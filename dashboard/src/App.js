import { useState } from "react";

const MAROON = "#8C1D40";
const GOLD = "#FFC627";
const BG = "#F8F9FA";
const CARD_BG = "#FFFFFF";
const TEXT = "#1A1A2E";
const MUTED = "#64748B";
const BORDER = "#E2E8F0";

const allResults = [
  { experiment: "E1 — Baseline", model: "Logistic Regression", precision: 0.9491, recall: 0.8664, f1: 0.9059, prauc: 0.9714 },
  { experiment: "E1 — Baseline", model: "Random Forest (Tuned)", precision: 0.9867, recall: 1.0000, f1: 0.9933, prauc: 0.9996 },
  { experiment: "E1 — Baseline", model: "XGBoost (Tuned)", precision: 0.9524, recall: 1.0000, f1: 0.9756, prauc: 0.9966 },
  { experiment: "E2 — Graph-Enhanced", model: "LR + Graph Features", precision: 0.9682, recall: 0.9740, f1: 0.9711, prauc: 0.9959 },
  { experiment: "E2 — Graph-Enhanced", model: "RF + Graph Features", precision: 0.9976, recall: 1.0000, f1: 0.9988, prauc: 1.0000 },
  { experiment: "E2 — Graph-Enhanced", model: "XGBoost + Graph Features", precision: 0.9792, recall: 1.0000, f1: 0.9895, prauc: 0.9993 },
  { experiment: "E3 — Embeddings", model: "GraphSAGE + RF", precision: 0.9963, recall: 1.0000, f1: 0.9981, prauc: 0.9997 },
  { experiment: "E3 — Embeddings", model: "GraphSAGE + MLP", precision: 0.9933, recall: 1.0000, f1: 0.9966, prauc: 0.9987 },
  { experiment: "E4 — Standalone GNN", model: "GraphSAGE", precision: 0.4344, recall: 0.9298, f1: 0.5922, prauc: 0.7143 },
  { experiment: "E5 — Clustering", model: "K-Means Purity", precision: null, recall: null, f1: null, prauc: 0.9916 },
];

const clusterData = [
  { cluster: "Cluster 4", total: 337, fraudRate: 99.7, type: "fraud" },
  { cluster: "Cluster 3", total: 8240, fraudRate: 61.4, type: "high" },
  { cluster: "Cluster 2", total: 72284, fraudRate: 1.7, type: "low" },
  { cluster: "Cluster 1", total: 74169, fraudRate: 0.8, type: "low" },
  { cluster: "Cluster 0", total: 439613, fraudRate: 0.0, type: "clean" },
];

const edaInsights = [
  { icon: "🏪", title: "Merchant Risk", value: "11/49", desc: "merchants have >50% fraud rate" },
  { icon: "🛍️", title: "Highest Risk Category", value: "es_leisure", desc: "80%+ fraud rate" },
  { icon: "💰", title: "Amount Correlation", value: "0.49", desc: "strongest single predictor" },
  { icon: "📉", title: "Fraud Rate Trend", value: "Decreasing", desc: "over time steps 0→179" },
  { icon: "👤", title: "Age Group Risk", value: "Group 0", desc: "highest fraud rate at ~2%" },
  { icon: "⚠️", title: "DBSCAN Anomalies", value: "65.2%", desc: "fraud in noise points" },
];

const tabs = ["Overview", "E1 Baseline", "E2 Graph", "E3 Embeddings", "E4 GNN", "E5 Clustering", "EDA Insights", "Results Gallery"];

const experimentColors = {
  "E1 — Baseline": "#3B82F6",
  "E2 — Graph-Enhanced": "#D97706",
  "E3 — Embeddings": "#059669",
  "E4 — Standalone GNN": "#F59E0B",
  "E5 — Clustering": "#7C3AED",
};

function MetricBadge({ value, highlight }) {
  const color = value >= 0.99 ? "#059669" : value >= 0.95 ? "#D97706" : value >= 0.80 ? "#F59E0B" : "#EF4444";
  return (
    <span style={{
      background: highlight ? `${color}18` : "transparent",
      color: highlight ? color : TEXT,
      padding: "2px 8px",
      borderRadius: 4,
      fontFamily: "monospace",
      fontSize: 13,
      fontWeight: highlight ? 700 : 400,
      border: highlight ? `1px solid ${color}44` : "none",
    }}>
      {value !== null ? value.toFixed(4) : "—"}
    </span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: CARD_BG,
      border: `1px solid ${BORDER}`,
      borderTop: `3px solid ${color}`,
      borderRadius: 12,
      padding: "20px 24px",
      flex: 1,
      minWidth: 140,
      boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
    }}>
      <div style={{ fontSize: 28, fontWeight: 800, color, fontFamily: "Georgia, serif", letterSpacing: -1 }}>{value}</div>
      <div style={{ fontSize: 13, color: TEXT, fontWeight: 600, marginTop: 4 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: MUTED, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
      <div style={{ width: 4, height: 24, background: GOLD, borderRadius: 2 }} />
      <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: TEXT, fontFamily: "Georgia, serif" }}>{children}</h2>
    </div>
  );
}

function ResultsTable({ rows }) {
  return (
    <div style={{ overflowX: "auto", background: CARD_BG, borderRadius: 12, border: `1px solid ${BORDER}`, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: MAROON }}>
            {["Experiment", "Model", "Precision", "Recall", "F1-Score", "PR-AUC"].map(h => (
              <th key={h} style={{ padding: "12px 16px", textAlign: "left", color: "#fff", fontWeight: 700, fontSize: 12, letterSpacing: 0.5 }}>{h.toUpperCase()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${BORDER}`, background: i % 2 === 0 ? "#F8F9FA" : CARD_BG }}>
              <td style={{ padding: "10px 16px", color: experimentColors[r.experiment] || MUTED, fontSize: 12, fontWeight: 600 }}>{r.experiment}</td>
              <td style={{ padding: "10px 16px", color: TEXT, fontWeight: 500 }}>{r.model}</td>
              <td style={{ padding: "10px 16px" }}><MetricBadge value={r.precision} /></td>
              <td style={{ padding: "10px 16px" }}><MetricBadge value={r.recall} /></td>
              <td style={{ padding: "10px 16px" }}><MetricBadge value={r.f1} /></td>
              <td style={{ padding: "10px 16px" }}><MetricBadge value={r.prauc} highlight /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PRBar({ value, label, experiment }) {
  const pct = (value / 1) * 100;
  const color = value >= 0.99 ? "#059669" : value >= 0.95 ? "#D97706" : value >= 0.80 ? "#F59E0B" : "#EF4444";
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontSize: 13, color: TEXT }}>{label}</span>
        <span style={{ fontSize: 11, color: experimentColors[experiment] || MUTED, fontWeight: 600 }}>{experiment}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ flex: 1, background: "#E2E8F0", borderRadius: 4, height: 10, overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4 }} />
        </div>
        <span style={{ fontSize: 12, color, fontFamily: "monospace", minWidth: 55, textAlign: "right", fontWeight: 700 }}>{value.toFixed(4)}</span>
      </div>
    </div>
  );
}

function ImageCard({ src, title }) {
  return (
    <div style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderRadius: 12, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
      <img src={src} alt={title} style={{ width: "100%", display: "block" }}
        onError={(e) => { e.target.style.display = "none"; }} />
      <div style={{ padding: "10px 14px", fontSize: 12, color: MUTED, fontWeight: 500 }}>{title}</div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0);

  const tabContent = [
    // OVERVIEW
    <div key="overview">
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 32 }}>
        <StatCard label="Total Transactions" value="594,643" sub="BankSim Dataset" color={GOLD} />
        <StatCard label="Fraud Cases" value="7,200" sub="1.21% of total" color={MAROON} />
        <StatCard label="Experiments Run" value="5" sub="E1 through E5" color="#3B82F6" />
        <StatCard label="Best PR-AUC" value="1.000" sub="RF + Graph Features" color="#059669" />
        <StatCard label="Graph Nodes" value="4,110" sub="Customers + Merchants" color="#7C3AED" />
        <StatCard label="Fraud Ring Found" value="99.7%" sub="K-Means Cluster 4" color="#F59E0B" />
      </div>

      <SectionTitle>Core Hypothesis Result</SectionTitle>
      <div style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderLeft: `4px solid #059669`, borderRadius: 12, padding: 24, marginBottom: 32, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
        <div style={{ fontSize: 15, color: TEXT, lineHeight: 1.7, marginBottom: 16 }}>
          <span style={{ color: "#059669", fontWeight: 700 }}>✓ Hypothesis Confirmed:</span> Graph-derived structure reveals fraud patterns that flat feature vectors miss. Every model improved when graph features were added.
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
          {[
            { label: "LR Recall Improvement", before: "0.87", after: "0.97", delta: "+0.10" },
            { label: "RF PR-AUC Improvement", before: "0.9996", after: "1.0000", delta: "+0.0004" },
            { label: "XGB PR-AUC Improvement", before: "0.9966", after: "0.9993", delta: "+0.0027" },
          ].map((item, i) => (
            <div key={i} style={{ background: "#F0FDF4", border: `1px solid #BBF7D0`, borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 11, color: MUTED, marginBottom: 8 }}>{item.label}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span style={{ color: MUTED, fontFamily: "monospace" }}>{item.before}</span>
                <span style={{ color: "#D97706" }}>→</span>
                <span style={{ color: "#059669", fontFamily: "monospace", fontWeight: 700 }}>{item.after}</span>
                <span style={{ background: "#DCFCE7", color: "#059669", padding: "2px 6px", borderRadius: 4, fontSize: 11, fontWeight: 700 }}>{item.delta}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <SectionTitle>All Experiments — PR-AUC Overview</SectionTitle>
      <div style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
        {allResults.map((r, i) => (
          <PRBar key={i} value={r.prauc} label={r.model} experiment={r.experiment} />
        ))}
      </div>
    </div>,

    // E1 BASELINE
    <div key="e1">
      <SectionTitle>E1 — Baseline Tabular ML Models</SectionTitle>
      <div style={{ background: "#EFF6FF", border: `1px solid #BFDBFE`, borderRadius: 12, padding: 20, marginBottom: 24 }}>
        <p style={{ color: MUTED, fontSize: 14, margin: 0 }}>Standard ML models trained on raw transaction features only. Each transaction is treated independently — no relationship information. Hyperparameter tuning applied using RandomizedSearchCV.</p>
      </div>
      <ResultsTable rows={allResults.filter(r => r.experiment === "E1 — Baseline")} />
      <div style={{ display: "flex", gap: 16, marginTop: 24, flexWrap: "wrap" }}>
        {[
          { model: "Logistic Regression", note: "Simple linear baseline. Good PR-AUC but lowest recall — misses ~13% of fraud.", color: "#3B82F6", bg: "#EFF6FF" },
          { model: "Random Forest (Tuned)", note: "Best performer. 200 trees, max_depth 20. Near perfect across all metrics.", color: "#059669", bg: "#F0FDF4" },
          { model: "XGBoost (Tuned)", note: "Strong gradient boosting. max_depth 9, learning_rate 0.2. Catches all fraud.", color: "#D97706", bg: "#FFFBEB" },
        ].map((item, i) => (
          <div key={i} style={{ flex: 1, minWidth: 200, background: item.bg, border: `1px solid ${item.color}33`, borderRadius: 10, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: item.color, marginBottom: 8 }}>{item.model}</div>
            <div style={{ fontSize: 12, color: MUTED, lineHeight: 1.6 }}>{item.note}</div>
          </div>
        ))}
      </div>
    </div>,

    // E2 GRAPH
    <div key="e2">
      <SectionTitle>E2 — Graph-Enhanced ML Models</SectionTitle>
      <div style={{ background: "#FFFBEB", border: `1px solid #FDE68A`, borderRadius: 12, padding: 20, marginBottom: 24 }}>
        <p style={{ color: MUTED, fontSize: 14, margin: "0 0 12px" }}>Transaction network built in Neo4j. Customers and merchants as nodes, transactions as weighted edges. 5 graph features extracted and merged with tabular data.</p>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {["Degree Centrality", "PageRank", "Betweenness Centrality", "Community Detection", "Merchant Fraud Rate"].map(f => (
            <span key={f} style={{ background: "#FEF3C7", color: "#D97706", padding: "4px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, border: "1px solid #FDE68A" }}>{f}</span>
          ))}
        </div>
      </div>
      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        <StatCard label="Customer Nodes" value="4,061" color="#D97706" />
        <StatCard label="Merchant Nodes" value="49" color="#D97706" />
        <StatCard label="Transaction Edges" value="50,000" color="#D97706" />
        <StatCard label="High Fraud Merchants" value="11/49" sub=">50% fraud rate" color={MAROON} />
      </div>
      <ResultsTable rows={allResults.filter(r => r.experiment === "E2 — Graph-Enhanced")} />
      <div style={{ background: "#FFF1F2", border: `1px solid ${MAROON}33`, borderRadius: 10, padding: 16, marginTop: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: MAROON, marginBottom: 6 }}>🔑 Key Finding</div>
        <div style={{ fontSize: 13, color: TEXT }}>11 out of 49 merchants have more than 50% fraud rate — completely invisible without graph analysis. The merchant fraud rate feature alone drives most of the improvement in recall.</div>
      </div>
    </div>,

    // E3 EMBEDDINGS
    <div key="e3">
      <SectionTitle>E3 — GraphSAGE Embedding-Based Models</SectionTitle>
      <div style={{ background: "#F0FDF4", border: `1px solid #BBF7D0`, borderRadius: 12, padding: 20, marginBottom: 24 }}>
        <p style={{ color: MUTED, fontSize: 14, margin: 0 }}>GraphSAGE learns its own node representations. Each customer and merchant gets a 32-dimensional fingerprint. Customer + Merchant fingerprints = 64 features per transaction fed into downstream classifiers.</p>
      </div>
      <ResultsTable rows={allResults.filter(r => r.experiment === "E3 — Embeddings")} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 24 }}>
        {[
          { model: "GraphSAGE + Random Forest", prauc: "0.9997", label: "Near Perfect" },
          { model: "GraphSAGE + MLP", prauc: "0.9987", label: "Excellent" },
        ].map((item, i) => (
          <div key={i} style={{ background: "#F0FDF4", border: `1px solid #BBF7D0`, borderRadius: 10, padding: 20 }}>
            <div style={{ fontSize: 13, color: MUTED, marginBottom: 4 }}>{item.model}</div>
            <div style={{ fontSize: 32, fontWeight: 800, color: "#059669", fontFamily: "Georgia, serif" }}>{item.prauc}</div>
            <div style={{ fontSize: 12, color: MUTED }}>PR-AUC — {item.label}</div>
          </div>
        ))}
      </div>
    </div>,

    // E4 GNN
    <div key="e4">
      <SectionTitle>E4 — Standalone GraphSAGE Classifier</SectionTitle>
      <div style={{ background: "#FFFBEB", border: `1px solid #FDE68A`, borderRadius: 12, padding: 20, marginBottom: 24 }}>
        <p style={{ color: MUTED, fontSize: 14, margin: 0 }}>End-to-end Graph Neural Network trained directly on the transaction graph. No manual feature engineering. 3 layers, 64 hidden units, 100 epochs.</p>
      </div>
      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        <StatCard label="Recall" value="0.9298" sub="Catches 93% of fraud" color="#059669" />
        <StatCard label="Precision" value="0.4344" sub="Higher false positives" color="#F59E0B" />
        <StatCard label="PR-AUC" value="0.7143" sub="Room for improvement" color={MAROON} />
        <StatCard label="F1-Score" value="0.5922" sub="Precision-recall tradeoff" color="#7C3AED" />
      </div>
      <div style={{ background: "#FFFBEB", border: `1px solid #FDE68A`, borderRadius: 10, padding: 16, marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#D97706", marginBottom: 6 }}>⚠️ Why Lower Performance?</div>
        <div style={{ fontSize: 13, color: TEXT, lineHeight: 1.7 }}>GraphSAGE as a standalone classifier underperforms due to limited node features (only 3), small graph size (4,061 nodes), and only 100 training epochs. With the GDS plugin for proper PageRank and extended training, performance would improve significantly.</div>
      </div>
      <div style={{ background: "#F0FDF4", border: `1px solid #BBF7D0`, borderRadius: 10, padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#059669", marginBottom: 6 }}>✓ Strong Recall</div>
        <div style={{ fontSize: 13, color: TEXT }}>Despite lower precision, the 93% recall means GraphSAGE catches almost all real fraud cases — valuable in banking where missing fraud is costly.</div>
      </div>
    </div>,

    // E5 CLUSTERING
    <div key="e5">
      <SectionTitle>E5 — Unsupervised Fraud Ring Detection</SectionTitle>
      <div style={{ background: "#F5F3FF", border: `1px solid #DDD6FE`, borderRadius: 12, padding: 20, marginBottom: 24 }}>
        <p style={{ color: MUTED, fontSize: 14, margin: 0 }}>K-Means and DBSCAN clustering applied to graph features WITHOUT fraud labels. Detects fraud rings purely through pattern recognition.</p>
      </div>
      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        <StatCard label="Silhouette Score" value="0.6315" sub="Good cluster quality" color="#7C3AED" />
        <StatCard label="Purity Score" value="0.9916" sub="Near perfect" color="#059669" />
        <StatCard label="DBSCAN Noise Fraud" value="65.2%" sub="vs 0.23% normal" color={MAROON} />
        <StatCard label="Pure Fraud Cluster" value="99.7%" sub="337 transactions" color="#F59E0B" />
      </div>
      <SectionTitle>K-Means Cluster Analysis</SectionTitle>
      <div style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderRadius: 12, overflow: "hidden", marginBottom: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: MAROON }}>
              {["Cluster", "Total Transactions", "Fraud Rate", "Classification"].map(h => (
                <th key={h} style={{ padding: "12px 16px", textAlign: "left", color: "#fff", fontWeight: 700, fontSize: 12 }}>{h.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {clusterData.map((c, i) => {
              const color = c.type === "fraud" ? "#EF4444" : c.type === "high" ? "#F59E0B" : c.type === "low" ? "#D97706" : "#059669";
              const label = c.type === "fraud" ? "🔴 Pure Fraud Ring" : c.type === "high" ? "🟠 High Risk" : c.type === "low" ? "🟡 Low Risk" : "🟢 Clean";
              return (
                <tr key={i} style={{ borderBottom: `1px solid ${BORDER}`, background: i % 2 === 0 ? "#F8F9FA" : CARD_BG }}>
                  <td style={{ padding: "12px 16px", color: TEXT, fontWeight: 600 }}>{c.cluster}</td>
                  <td style={{ padding: "12px 16px", color: MUTED, fontFamily: "monospace" }}>{c.total.toLocaleString()}</td>
                  <td style={{ padding: "12px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, background: "#E2E8F0", borderRadius: 4, height: 8, maxWidth: 120, overflow: "hidden" }}>
                        <div style={{ width: `${c.fraudRate}%`, height: "100%", background: color, borderRadius: 4 }} />
                      </div>
                      <span style={{ color, fontFamily: "monospace", fontWeight: 700 }}>{c.fraudRate}%</span>
                    </div>
                  </td>
                  <td style={{ padding: "12px 16px", color, fontWeight: 600 }}>{label}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ background: "#FFF1F2", border: `1px solid ${MAROON}33`, borderRadius: 10, padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: MAROON, marginBottom: 6 }}>🔑 Key Finding</div>
        <div style={{ fontSize: 13, color: TEXT, lineHeight: 1.7 }}>DBSCAN identified anomalous points with 65.2% fraud rate vs 0.23% in normal clusters — proving unsupervised methods can detect fraud rings without ever using fraud labels.</div>
      </div>
    </div>,

    // EDA INSIGHTS
    <div key="eda">
      <SectionTitle>Exploratory Data Analysis — Key Insights</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16, marginBottom: 32 }}>
        {edaInsights.map((insight, i) => (
          <div key={i} style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 20, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>{insight.icon}</div>
            <div style={{ fontSize: 12, color: MUTED, marginBottom: 4 }}>{insight.title}</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: MAROON, fontFamily: "Georgia, serif", marginBottom: 4 }}>{insight.value}</div>
            <div style={{ fontSize: 12, color: MUTED }}>{insight.desc}</div>
          </div>
        ))}
      </div>
      <SectionTitle>Fraud by Transaction Category</SectionTitle>
      <div style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 24, marginBottom: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
        {[
          { cat: "es_leisure", rate: 87 },
          { cat: "es_travel", rate: 79 },
          { cat: "es_sportsandtoys", rate: 47 },
          { cat: "es_hotelservices", rate: 30 },
          { cat: "es_otherservices", rate: 24 },
          { cat: "es_home", rate: 15 },
          { cat: "es_health", rate: 9 },
          { cat: "es_wellnessandbeauty", rate: 5 },
        ].map((item, i) => (
          <div key={i} style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 13, color: TEXT, fontFamily: "monospace" }}>{item.cat}</span>
              <span style={{ fontSize: 12, color: item.rate > 50 ? "#EF4444" : item.rate > 20 ? "#F59E0B" : "#D97706", fontWeight: 700 }}>{item.rate}%</span>
            </div>
            <div style={{ background: "#E2E8F0", borderRadius: 4, height: 10, overflow: "hidden" }}>
              <div style={{ width: `${item.rate}%`, height: "100%", background: item.rate > 50 ? "#EF4444" : item.rate > 20 ? "#F59E0B" : "#D97706", borderRadius: 4 }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ background: CARD_BG, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: TEXT, marginBottom: 16 }}>Dataset Statistics</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
          {[
            { label: "Total Transactions", value: "594,643" },
            { label: "Fraudulent", value: "7,200 (1.21%)" },
            { label: "Legitimate", value: "587,443 (98.79%)" },
            { label: "Imbalance Ratio", value: "1:81" },
            { label: "Time Steps", value: "0 → 179" },
            { label: "Unique Customers", value: "~4,000+" },
            { label: "Unique Merchants", value: "49" },
            { label: "Transaction Categories", value: "15" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: `1px solid ${BORDER}` }}>
              <span style={{ fontSize: 13, color: MUTED }}>{item.label}</span>
              <span style={{ fontSize: 13, color: TEXT, fontWeight: 600, fontFamily: "monospace" }}>{item.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>,

    // RESULTS GALLERY
    <div key="gallery">
      <SectionTitle>Results Gallery — All Plots</SectionTitle>

      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: MAROON, marginBottom: 16, paddingBottom: 8, borderBottom: `2px solid ${GOLD}` }}>Baseline Models</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <ImageCard src="/images/pr_curves_baseline.png" title="PR Curves — Baseline Models" />
          <ImageCard src="/images/model_comparison.png" title="Model Comparison" />
          <ImageCard src="/images/confusion_matrices.png" title="Confusion Matrices" />
        </div>
      </div>

      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: MAROON, marginBottom: 16, paddingBottom: 8, borderBottom: `2px solid ${GOLD}` }}>Graph-Enhanced Models</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <ImageCard src="/images/baseline_vs_hybrid.png" title="Baseline vs Graph-Enhanced" />
          <ImageCard src="/images/pr_curves_comparison.png" title="PR Curves Comparison" />
        </div>
      </div>

      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: MAROON, marginBottom: 16, paddingBottom: 8, borderBottom: `2px solid ${GOLD}` }}>GraphSAGE</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <ImageCard src="/images/graphsage_loss.png" title="GraphSAGE Training Loss" />
          <ImageCard src="/images/graphsage_pr_curve.png" title="GraphSAGE PR Curve" />
          <ImageCard src="/images/e3_embedding_pr_curves.png" title="E3 Embedding Models PR Curves" />
          <ImageCard src="/images/full_comparison.png" title="Full Model Comparison" />
        </div>
      </div>

      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: MAROON, marginBottom: 16, paddingBottom: 8, borderBottom: `2px solid ${GOLD}` }}>Clustering Analysis</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <ImageCard src="/images/clustering_results.png" title="Clustering Results" />
        </div>
      </div>

      <div>
        <div style={{ fontSize: 14, fontWeight: 700, color: MAROON, marginBottom: 16, paddingBottom: 8, borderBottom: `2px solid ${GOLD}` }}>EDA Visualizations</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <ImageCard src="/images/01_class_distribution.png" title="Class Distribution" />
          <ImageCard src="/images/02_fraud_by_category.png" title="Fraud by Category" />
          <ImageCard src="/images/03_amount_distribution.png" title="Amount Distribution" />
          <ImageCard src="/images/04_fraud_over_time.png" title="Fraud Over Time" />
          <ImageCard src="/images/05_fraud_by_demographics.png" title="Fraud by Demographics" />
          <ImageCard src="/images/06_merchant_fraud.png" title="Merchant Fraud Analysis" />
          <ImageCard src="/images/07_correlation_heatmap.png" title="Correlation Heatmap" />
        </div>
      </div>
    </div>,
  ];

  return (
    <div style={{ minHeight: "100vh", background: BG, color: TEXT, fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{ background: MAROON, padding: "20px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 11, color: GOLD, fontWeight: 700, letterSpacing: 2, marginBottom: 4 }}>CSE 573 — SEMANTIC WEB MINING | GROUP 23 | PROJECT 10</div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: "#fff", fontFamily: "Georgia, serif" }}>
            Fraud Detection with Graph Databases & GNNs
          </h1>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: `${GOLD}99`, marginBottom: 2 }}>Arizona State University</div>
          <div style={{ fontSize: 11, color: "#ffffff88" }}>BankSim Dataset · 594,643 Transactions</div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: CARD_BG, borderBottom: `1px solid ${BORDER}`, padding: "0 32px", display: "flex", gap: 4, overflowX: "auto", boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>
        {tabs.map((tab, i) => (
          <button key={i} onClick={() => setActiveTab(i)} style={{
            background: "none",
            border: "none",
            borderBottom: activeTab === i ? `3px solid ${MAROON}` : "3px solid transparent",
            color: activeTab === i ? MAROON : MUTED,
            padding: "14px 18px",
            cursor: "pointer",
            fontSize: 13,
            fontWeight: activeTab === i ? 700 : 400,
            whiteSpace: "nowrap",
            transition: "all 0.2s",
          }}>
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "32px", maxWidth: 1100, margin: "0 auto" }}>
        {tabContent[activeTab]}
      </div>

      {/* Footer */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center", background: CARD_BG }}>
        <div style={{ fontSize: 11, color: MUTED }}>
          Luan Nguyen · Dhanush M S · Adrian Zhang · Shashikant Nanda · Chandra Pavuluri · Chitwandeep Kaur
        </div>
        <a href="https://github.com/LuaanNguyen/fraud-detection" target="_blank" rel="noreferrer"
          style={{ fontSize: 11, color: MAROON, textDecoration: "none", fontWeight: 600 }}>
          GitHub Repository →
        </a>
      </div>
    </div>
  );
}