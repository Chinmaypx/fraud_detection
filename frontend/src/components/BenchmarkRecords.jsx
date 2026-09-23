function format(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(4);
}

function metricRow(metrics = {}) {
  return {
    'ROC-AUC': metrics['ROC-AUC'],
    'PR-AUC': metrics['PR-AUC'],
    Precision: metrics.Precision,
    Recall: metrics['Recall (Sensitivity)'],
    F1: metrics['F1 Score'],
  };
}

export default function BenchmarkRecords({ mlpMetrics, mlpFeatures, ieeeInfo }) {
  const ieeeReport = ieeeInfo?.metrics || {};
  const ieeeValidation = ieeeReport.validation?.metrics || {};
  const ieeeTest = ieeeReport.test?.metrics || {};
  const ieeeMatrix = ieeeReport.test?.confusion_matrix;
  const records = [
    {
      dataset: 'Generated banking transaction data · held-out test',
      model: 'MLP Model · FraudDetectorNet',
      features: mlpFeatures?.length
        ? `${mlpFeatures.length} features: ${mlpFeatures.join(', ')}`
        : 'Engineered transaction features',
      metrics: metricRow(mlpMetrics || {}),
    },
    {
      dataset: 'IEEE-CIS Fraud Detection · chronological held-out test',
      model: 'IEEE-CIS Fraud Model · FraudDetectorNet',
      features: ieeeInfo?.features_used?.join(', ') || 'IEEE-CIS user-facing features',
      metrics: metricRow(ieeeTest),
    },
  ];

  return (
    <>
      <section className="card animate-in" aria-labelledby="benchmark-records-heading">
        <div className="card-header">
          <span className="card-title" id="benchmark-records-heading">Benchmark Records</span>
          <span className="model-tag mlp">Separate datasets</span>
        </div>
        <p className="benchmark-note">
          The original MLP uses generated transaction data. IEEE-CIS results are reported separately;
          these records are not directly ranked against each other.
        </p>
        <div className="table-responsive">
          <table className="metrics-table benchmark-records-table">
            <thead><tr>
              <th>Dataset</th><th>Model</th><th>Features</th>
              <th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th>
            </tr></thead>
            <tbody>{records.map((record) => (
              <tr key={record.dataset}>
                <td>{record.dataset}</td><td>{record.model}</td><td className="benchmark-features">{record.features}</td>
                {Object.values(record.metrics).map((value, index) => <td key={index}>{format(value)}</td>)}
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>

      <section className="card animate-in" aria-labelledby="ieee-benchmark-heading" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title" id="ieee-benchmark-heading">IEEE-CIS Fraud Model Benchmark</span>
          <span className="model-tag mlp">Validation and held-out test</span>
        </div>
        <p className="benchmark-note">
          Chronological train, validation, and test partitions. The decision threshold is selected using validation data;
          held-out test metrics are calculated after model selection.
        </p>
        <div className="grid-2">
          {[["Validation", ieeeValidation], ["Held-out test", ieeeTest]].map(([label, metrics]) => (
            <div className="card benchmark-split-card" key={label}>
              <h3>{label}</h3>
              <dl className="benchmark-metrics">
                {Object.entries(metricRow(metrics)).map(([name, value]) => (
                  <div key={name}><dt>{name}</dt><dd>{format(value)}</dd></div>
                ))}
              </dl>
              {label === 'Held-out test' && ieeeMatrix && (
                <div className="benchmark-confusion">
                  <strong>Test confusion matrix</strong>
                  <span>TN {Number(ieeeMatrix[0][0]).toLocaleString()} · FP {Number(ieeeMatrix[0][1]).toLocaleString()} · FN {Number(ieeeMatrix[1][0]).toLocaleString()} · TP {Number(ieeeMatrix[1][1]).toLocaleString()}</span>
                </div>
              )}
            </div>
          ))}
        </div>
        {ieeeInfo?.threshold != null && <p className="benchmark-note">Validation-selected decision threshold: {Number(ieeeInfo.threshold).toFixed(6)}</p>}
      </section>
    </>
  );
}
