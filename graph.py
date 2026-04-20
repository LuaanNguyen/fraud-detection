"""
graph.py
---------
Builds a transaction graph in Neo4j from BankSim data.
Customers and Merchants are nodes, Transactions are edges.
Extracts graph features: Degree, PageRank, Betweenness Centrality.
"""

import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from preprocessing import load_data

# ---------------------------------------------------------------
# Neo4j Connection (override via environment variables)
# ---------------------------------------------------------------
NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class FraudGraph:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        print("[INFO] Connected to Neo4j")

    def close(self):
        self.driver.close()

    # ---------------------------------------------------------------
    # 1. Clear existing data
    # ---------------------------------------------------------------
    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[INFO] Cleared existing graph data")

    # ---------------------------------------------------------------
    # 2. Load data into Neo4j
    # ---------------------------------------------------------------
    def build_graph(self, df, sample_size=50000):
        """
        Load transactions into Neo4j as a graph.
        Customers and Merchants = Nodes
        Transactions = Edges
        Uses a sample to keep it fast.
        """
        print(f"\n[INFO] Building graph with {sample_size:,} transactions...")

        # Sample for performance
        df_sample = df.sample(n=sample_size, random_state=42)

        # Create constraints for uniqueness
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE")

        # Batch insert
        batch_size = 1000
        records = df_sample.to_dict("records")

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            with self.driver.session() as session:
                session.run("""
                    UNWIND $batch AS row
                    MERGE (c:Customer {id: row.customer})
                      SET c.age      = row.age,
                          c.gender   = row.gender
                    MERGE (m:Merchant {id: row.merchant})
                      SET m.category = row.category
                    CREATE (c)-[:TRANSACTION {
                        amount   : row.amount,
                        category : row.category,
                        step     : row.step,
                        fraud    : row.fraud
                    }]->(m)
                """, batch=batch)

            if (i // batch_size) % 10 == 0:
                print(f"  Inserted {min(i + batch_size, len(records)):,} / {len(records):,} records")

        print(f"[INFO] Graph built successfully!")
        self._print_stats()

    # ---------------------------------------------------------------
    # 3. Print graph statistics
    # ---------------------------------------------------------------
    def _print_stats(self):
        with self.driver.session() as session:
            customers  = session.run("MATCH (c:Customer) RETURN count(c) AS count").single()["count"]
            merchants  = session.run("MATCH (m:Merchant) RETURN count(m) AS count").single()["count"]
            txns       = session.run("MATCH ()-[t:TRANSACTION]->() RETURN count(t) AS count").single()["count"]
            fraud_txns = session.run("MATCH ()-[t:TRANSACTION {fraud:1}]->() RETURN count(t) AS count").single()["count"]

        print(f"\n{'='*50}")
        print(f"  GRAPH STATISTICS")
        print(f"{'='*50}")
        print(f"  Customer Nodes  : {customers:,}")
        print(f"  Merchant Nodes  : {merchants:,}")
        print(f"  Transactions    : {txns:,}")
        print(f"  Fraud Edges     : {fraud_txns:,}")
        print(f"{'='*50}")

    # ---------------------------------------------------------------
    # 4. Extract Degree Centrality
    # ---------------------------------------------------------------
    def extract_degree_centrality(self, df):
        """Count how many transactions each customer and merchant has."""
        print("\n[INFO] Extracting Degree Centrality...")

        customer_degree = df.groupby("customer").size().reset_index()
        customer_degree.columns = ["customer", "customer_degree"]

        merchant_degree = df.groupby("merchant").size().reset_index()
        merchant_degree.columns = ["merchant", "merchant_degree"]

        df = df.merge(customer_degree, on="customer", how="left")
        df = df.merge(merchant_degree, on="merchant", how="left")

        print(f"  Customer degree — mean: {df['customer_degree'].mean():.2f}, max: {df['customer_degree'].max()}")
        print(f"  Merchant degree — mean: {df['merchant_degree'].mean():.2f}, max: {df['merchant_degree'].max()}")

        return df

    # ---------------------------------------------------------------
    # 5. Extract PageRank (via Neo4j GDS)
    # ---------------------------------------------------------------
    def extract_pagerank(self, df):
        """Run PageRank on the graph using Neo4j."""
        print("\n[INFO] Extracting PageRank...")

        try:
            with self.driver.session() as session:
                # Project graph
                session.run("""
                    CALL gds.graph.project(
                        'fraudGraph',
                        ['Customer', 'Merchant'],
                        {TRANSACTION: {orientation: 'NATURAL'}}
                    )
                """)

                # Run PageRank
                result = session.run("""
                    CALL gds.pageRank.stream('fraudGraph')
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS nodeId, score
                    ORDER BY score DESC
                """)
                pr_data = {row["nodeId"]: row["score"] for row in result}

                # Drop projected graph
                session.run("CALL gds.graph.drop('fraudGraph')")

            # Map back to dataframe
            df["customer_pagerank"] = df["customer"].map(pr_data).fillna(0)
            df["merchant_pagerank"] = df["merchant"].map(pr_data).fillna(0)

            print(f"  PageRank extracted for {len(pr_data):,} nodes")

        except Exception as e:
            print(f"  [WARNING] PageRank failed (GDS plugin may not be installed): {e}")
            print("  [INFO] Using degree centrality as PageRank proxy instead")
            df["customer_pagerank"] = df.get("customer_degree", 0)
            df["merchant_pagerank"] = df.get("merchant_degree", 0)

        return df

    # ---------------------------------------------------------------
    # 6. Extract Betweenness Centrality
    # ---------------------------------------------------------------
    def extract_betweenness_centrality(self, df):
        """Approximate betweenness centrality for customers and merchants.

        Uses pandas groupby to compute a proxy: for each node, the fraction
        of unique counterparties it connects to out of all possible
        counterparties.  When Neo4j GDS is available, runs the real
        algorithm instead.
        """
        print("\n[INFO] Extracting Betweenness Centrality...")

        try:
            with self.driver.session() as session:
                session.run("""
                    CALL gds.graph.project(
                        'betweennessGraph',
                        ['Customer', 'Merchant'],
                        {TRANSACTION: {orientation: 'UNDIRECTED'}}
                    )
                """)
                result = session.run("""
                    CALL gds.betweenness.stream('betweennessGraph')
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS nodeId, score
                """)
                bc_data = {row["nodeId"]: row["score"] for row in result}
                session.run("CALL gds.graph.drop('betweennessGraph')")

            df["customer_betweenness"] = df["customer"].map(bc_data).fillna(0)
            df["merchant_betweenness"] = df["merchant"].map(bc_data).fillna(0)
            print(f"  Betweenness centrality extracted for {len(bc_data):,} nodes (GDS)")

        except Exception as e:
            print(f"  [WARNING] GDS betweenness failed: {e}")
            print("  [INFO] Computing pandas-based proxy instead")

            n_merchants = df["merchant"].nunique()
            n_customers = df["customer"].nunique()

            cust_uniq = df.groupby("customer")["merchant"].nunique().reset_index()
            cust_uniq.columns = ["customer", "customer_betweenness"]
            cust_uniq["customer_betweenness"] = cust_uniq["customer_betweenness"] / max(n_merchants, 1)

            merch_uniq = df.groupby("merchant")["customer"].nunique().reset_index()
            merch_uniq.columns = ["merchant", "merchant_betweenness"]
            merch_uniq["merchant_betweenness"] = merch_uniq["merchant_betweenness"] / max(n_customers, 1)

            df = df.merge(cust_uniq, on="customer", how="left")
            df = df.merge(merch_uniq, on="merchant", how="left")

        print(f"  Customer betweenness — mean: {df['customer_betweenness'].mean():.4f}, max: {df['customer_betweenness'].max():.4f}")
        print(f"  Merchant betweenness — mean: {df['merchant_betweenness'].mean():.4f}, max: {df['merchant_betweenness'].max():.4f}")

        return df

    # ---------------------------------------------------------------
    # 7. Community Detection (Louvain)
    # ---------------------------------------------------------------
    def extract_community_ids(self, df):
        """Assign community IDs to customers and merchants.

        Uses Neo4j GDS Louvain when available, otherwise falls back to a
        pandas-based heuristic that groups merchants by their dominant
        transaction category and customers by their most-used merchant
        community.
        """
        print("\n[INFO] Extracting Community Detection (Louvain)...")

        try:
            with self.driver.session() as session:
                session.run("""
                    CALL gds.graph.project(
                        'communityGraph',
                        ['Customer', 'Merchant'],
                        {TRANSACTION: {orientation: 'UNDIRECTED'}}
                    )
                """)
                result = session.run("""
                    CALL gds.louvain.stream('communityGraph')
                    YIELD nodeId, communityId
                    RETURN gds.util.asNode(nodeId).id AS nodeId, communityId
                """)
                comm_data = {row["nodeId"]: row["communityId"] for row in result}
                session.run("CALL gds.graph.drop('communityGraph')")

            df["customer_community"] = df["customer"].map(comm_data).fillna(-1).astype(int)
            df["merchant_community"] = df["merchant"].map(comm_data).fillna(-1).astype(int)
            n_communities = len(set(comm_data.values()))
            print(f"  Louvain detected {n_communities} communities (GDS)")

        except Exception as e:
            print(f"  [WARNING] GDS Louvain failed: {e}")
            print("  [INFO] Computing pandas-based community proxy instead")

            if "category" in df.columns:
                cat_col = "category"
            else:
                cat_col = None

            if cat_col:
                merch_comm = df.groupby("merchant")[cat_col].agg(
                    lambda x: x.value_counts().index[0]
                ).reset_index()
                merch_comm.columns = ["merchant", "merchant_community"]
                merch_comm["merchant_community"] = pd.factorize(merch_comm["merchant_community"])[0]

                df = df.merge(merch_comm, on="merchant", how="left")

                cust_comm = df.groupby("customer")["merchant_community"].agg(
                    lambda x: x.value_counts().index[0]
                ).reset_index()
                cust_comm.columns = ["customer", "customer_community"]
                df = df.merge(cust_comm, on="customer", how="left")
            else:
                df["customer_community"] = 0
                df["merchant_community"] = 0

        n_cust_comm = df["customer_community"].nunique()
        n_merch_comm = df["merchant_community"].nunique()
        print(f"  Customer communities: {n_cust_comm}")
        print(f"  Merchant communities: {n_merch_comm}")

        return df

    # ---------------------------------------------------------------
    # 8. Extract Fraud Rate per Merchant
    # ---------------------------------------------------------------
    def extract_merchant_fraud_rate(self, df, reference_df=None):
        """Calculate fraud rate per merchant.

        Parameters
        ----------
        df           : DataFrame to enrich with the new column.
        reference_df : If provided, fraud rates are computed from this
                       DataFrame only (use the training set to avoid
                       data leakage). Unseen merchants get rate 0.
        """
        print("\n[INFO] Extracting Merchant Fraud Rate...")

        source = reference_df if reference_df is not None else df

        merchant_fraud = source.groupby("merchant")["fraud"].agg(["sum", "count"]).reset_index()
        merchant_fraud.columns = ["merchant", "merchant_fraud_count", "merchant_txn_count"]
        merchant_fraud["merchant_fraud_rate"] = (
            merchant_fraud["merchant_fraud_count"] / merchant_fraud["merchant_txn_count"]
        )

        df = df.merge(merchant_fraud[["merchant", "merchant_fraud_rate"]], on="merchant", how="left")
        df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(0)

        print(f"  Merchants with >50% fraud rate: {(merchant_fraud['merchant_fraud_rate'] > 0.5).sum()}")
        print(f"  Merchants with 0% fraud rate  : {(merchant_fraud['merchant_fraud_rate'] == 0).sum()}")

        return df

    # ---------------------------------------------------------------
    # 9. Full graph feature extraction pipeline
    # ---------------------------------------------------------------
    def extract_graph_features(self, df):
        """Run all graph feature extractions and return enriched dataframe."""
        print(f"\n{'='*50}")
        print("  GRAPH FEATURE EXTRACTION")
        print(f"{'='*50}")

        df = self.extract_degree_centrality(df)
        df = self.extract_pagerank(df)
        df = self.extract_betweenness_centrality(df)
        df = self.extract_community_ids(df)
        df = self.extract_merchant_fraud_rate(df)

        graph_features = [
            "customer", "merchant", "fraud",
            "customer_degree", "merchant_degree",
            "customer_pagerank", "merchant_pagerank",
            "customer_betweenness", "merchant_betweenness",
            "customer_community", "merchant_community",
            "merchant_fraud_rate",
        ]
        out_path = os.path.join(RESULTS_DIR, "graph_features.csv")
        df[graph_features].to_csv(out_path, index=False)
        print(f"\n  [Saved] Graph features → {out_path}")

        return df


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def run_graph_pipeline():
    # Load raw data
    df = load_data()

    # Build graph
    fg = FraudGraph()
    fg.clear_graph()
    fg.build_graph(df, sample_size=50000)

    # Extract features on full dataset
    df = fg.extract_graph_features(df)

    fg.close()
    print("\n[INFO] Graph pipeline complete!")
    return df


if __name__ == "__main__":
    run_graph_pipeline()