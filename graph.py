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
# Neo4j Connection
# ---------------------------------------------------------------
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"  

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
    # 6. Extract Fraud Rate per Merchant
    # ---------------------------------------------------------------
    def extract_merchant_fraud_rate(self, df):
        """Calculate fraud rate per merchant — highly informative feature."""
        print("\n[INFO] Extracting Merchant Fraud Rate...")

        merchant_fraud = df.groupby("merchant")["fraud"].agg(["sum", "count"]).reset_index()
        merchant_fraud.columns = ["merchant", "merchant_fraud_count", "merchant_txn_count"]
        merchant_fraud["merchant_fraud_rate"] = (
            merchant_fraud["merchant_fraud_count"] / merchant_fraud["merchant_txn_count"]
        )

        df = df.merge(merchant_fraud[["merchant", "merchant_fraud_rate"]], on="merchant", how="left")

        print(f"  Merchants with >50% fraud rate: {(merchant_fraud['merchant_fraud_rate'] > 0.5).sum()}")
        print(f"  Merchants with 0% fraud rate  : {(merchant_fraud['merchant_fraud_rate'] == 0).sum()}")

        return df

    # ---------------------------------------------------------------
    # 7. Full graph feature extraction pipeline
    # ---------------------------------------------------------------
    def extract_graph_features(self, df):
        """Run all graph feature extractions and return enriched dataframe."""
        print(f"\n{'='*50}")
        print("  GRAPH FEATURE EXTRACTION")
        print(f"{'='*50}")

        df = self.extract_degree_centrality(df)
        df = self.extract_pagerank(df)
        df = self.extract_merchant_fraud_rate(df)

        # Save enriched features
        graph_features = [
            "customer", "merchant", "fraud",
            "customer_degree", "merchant_degree",
            "customer_pagerank", "merchant_pagerank",
            "merchant_fraud_rate"
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