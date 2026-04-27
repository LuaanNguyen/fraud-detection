"""
graph.py
---------
Builds a transaction graph in Neo4j from BankSim data.
Customers and Merchants are nodes, Transactions are edges.
Extracts graph features: Degree, PageRank, Betweenness Centrality,
Community Detection, Merchant Fraud Rate.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "EVALUATIONS")
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
        Transactions = Edges (weighted by amount)
        Multiple transactions between same entities kept as separate edges.
        """
        print(f"\n[INFO] Building graph with {sample_size:,} transactions...")
        print("  Edge weights    : transaction amount")
        print("  Multi-edges     : kept as separate time-ordered edges")

        df_sample = df.sample(n=sample_size, random_state=42)

        # Create constraints
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
                        fraud    : row.fraud,
                        weight   : row.amount
                    }]->(m)
                """, batch=batch)

            if (i // batch_size) % 10 == 0:
                print(f"  Inserted {min(i + batch_size, len(records)):,} / {len(records):,} records")

        print("[INFO] Graph built successfully!")
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
    # 5. Extract PageRank (via Neo4j GDS or proxy)
    # ---------------------------------------------------------------
    def extract_pagerank(self, df):
        """Run PageRank on the graph using Neo4j."""
        print("\n[INFO] Extracting PageRank...")

        try:
            with self.driver.session() as session:
                session.run("""
                    CALL gds.graph.project(
                        'fraudGraph',
                        ['Customer', 'Merchant'],
                        {TRANSACTION: {orientation: 'NATURAL'}}
                    )
                """)

                result = session.run("""
                    CALL gds.pageRank.stream('fraudGraph')
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS nodeId, score
                    ORDER BY score DESC
                """)
                pr_data = {row["nodeId"]: row["score"] for row in result}
                session.run("CALL gds.graph.drop('fraudGraph')")

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
        """
        Betweenness Centrality measures how often a node appears
        on the shortest path between other nodes.
        High betweenness = potential fraud bridge/hub.
        Computed using networkx since GDS plugin is not available.
        """
        print("\n[INFO] Extracting Betweenness Centrality...")

        try:
            import networkx as nx

            # Build graph from sample for performance
            G = nx.Graph()

            # Add edges
            for _, row in df.iterrows():
                G.add_edge(
                    f"C_{row['customer']}",
                    f"M_{row['merchant']}",
                    weight=row["amount"]
                )

            print(f"  Graph nodes: {G.number_of_nodes():,}")
            print(f"  Graph edges: {G.number_of_edges():,}")

            # Compute betweenness centrality
            # k=500 means we use 500 sample nodes for approximation (faster)
            print("  Computing betweenness centrality (approximate)...")
            bc = nx.betweenness_centrality(G, k=min(500, len(G.nodes)), weight="weight")

            # Map back to customers and merchants
            customer_bc = {
                k.replace("C_", ""): v
                for k, v in bc.items() if k.startswith("C_")
            }
            merchant_bc = {
                k.replace("M_", ""): v
                for k, v in bc.items() if k.startswith("M_")
            }

            df["customer_betweenness"] = df["customer"].map(customer_bc).fillna(0)
            df["merchant_betweenness"] = df["merchant"].map(merchant_bc).fillna(0)

            print(f"  Customer betweenness — mean: {df['customer_betweenness'].mean():.6f}")
            print(f"  Merchant betweenness — mean: {df['merchant_betweenness'].mean():.6f}")

        except ImportError:
            print("  [WARNING] networkx not installed. Installing...")
            os.system("pip install networkx")
            df["customer_betweenness"] = 0
            df["merchant_betweenness"] = 0

        except Exception as e:
            print(f"  [WARNING] Betweenness centrality failed: {e}")
            df["customer_betweenness"] = 0
            df["merchant_betweenness"] = 0

        return df

    # ---------------------------------------------------------------
    # 7. Extract Community Detection
    # ---------------------------------------------------------------
    def extract_community_detection(self, df):
        """
        Community Detection groups customers and merchants into
        clusters based on their transaction patterns.
        Fraud often concentrates in specific communities.
        Uses Louvain method via networkx.
        """
        print("\n[INFO] Extracting Community Detection...")

        try:
            import networkx as nx
            from collections import defaultdict

            # Build graph
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_edge(
                    f"C_{row['customer']}",
                    f"M_{row['merchant']}",
                    weight=row["amount"]
                )

            # Use connected components as community proxy
            # (Louvain requires python-louvain package)
            print("  Using connected components as community labels...")
            communities = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    communities[node] = i

            total_communities = len(set(communities.values()))
            print(f"  Total communities found: {total_communities:,}")

            # Map back
            customer_community = {
                k.replace("C_", ""): v
                for k, v in communities.items() if k.startswith("C_")
            }
            merchant_community = {
                k.replace("M_", ""): v
                for k, v in communities.items() if k.startswith("M_")
            }

            df["customer_community"] = df["customer"].map(customer_community).fillna(-1).astype(int)
            df["merchant_community"] = df["merchant"].map(merchant_community).fillna(-1).astype(int)

            # Compute fraud rate per community
            community_fraud = df.groupby("customer_community")["fraud"].mean().reset_index()
            community_fraud.columns = ["customer_community", "community_fraud_rate"]
            df = df.merge(community_fraud, on="customer_community", how="left")

            print(f"  Communities with >50% fraud rate: {(community_fraud['community_fraud_rate'] > 0.5).sum()}")

        except Exception as e:
            print(f"  [WARNING] Community detection failed: {e}")
            df["customer_community"] = 0
            df["merchant_community"] = 0
            df["community_fraud_rate"] = 0

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
        df = self.extract_community_detection(df)
        df = self.extract_merchant_fraud_rate(df)

        # Save enriched features
        graph_features = [
            "customer", "merchant", "fraud",
            "customer_degree", "merchant_degree",
            "customer_pagerank", "merchant_pagerank",
            "customer_betweenness", "merchant_betweenness",
            "customer_community", "merchant_community",
            "community_fraud_rate", "merchant_fraud_rate"
        ]

        # Only keep columns that exist
        graph_features = [c for c in graph_features if c in df.columns]

        out_path = os.path.join(RESULTS_DIR, "graph_features.csv")
        df[graph_features].to_csv(out_path, index=False)
        print(f"\n  [Saved] Graph features → {out_path}")
        print(f"  Total graph features: {len(graph_features) - 3}")

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