const ulbMetrics = {
  validation: {
    Precision: 0.8750,
    Recall: 0.7119,
    F1: 0.7850,
    'ROC-AUC': 0.9334,
    'PR-AUC': 0.6933,
  },
  test: {
    Precision: 0.9815,
    Recall: 0.6974,
    F1: 0.8154,
    'ROC-AUC': 0.9694,
    'PR-AUC': 0.8073,
  },
};

function format(value) {
  return value == null ? '—' : Number(value).toFixed(4);
}

export function ULBMetricDetails() {
  return (
    <section className="card animate-in" aria-labelledby="ulb-metrics-heading">
      <div className="card-header">
        <span className="card-title" id="ulb-metrics-heading">ULB Real-World Benchmark</span>
        <span className="model-tag mlp">Held-out test</span>
      </div>
      <p className="benchmark-note">
        ULB/Worldline credit-card fraud benchmark. Validation metrics use the validation split;
        held-out test metrics and confusion matrix use the separate test split.
      </p>
      <div className="grid-2">
        {['validation', 'test'].map((split) => (
          <div className="card benchmark-split-card" key={split}>
            <h3>{split === 'test' ? 'Held-out test' : 'Validation'}</h3>
            <dl className="benchmark-metrics">
              {Object.entries(ulbMetrics[split]).map(([name, value]) => (
                <div key={name}>
                  <dt>{name}</dt><dd>{format(value)}</dd>
                </div>
              ))}
            </dl>
            {split === 'test' && (
              <div className="benchmark-confusion">
                <strong>Test confusion matrix</strong>
                <span>TN 59,811 · FP 1 · FN 23 · TP 53</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

export default function BenchmarkRecords({ syntheticMetrics, syntheticFeatures }) {
  const records = [
    {
      dataset: 'Synthetic generated transactions (held-out test)',
      model: 'Synthetic MLP · FraudDetectorNet',
      features: syntheticFeatures?.length
        ? `${syntheticFeatures.length} features: ${syntheticFeatures.join(', ')}`
        : 'Synthetic engineered transaction features',
      metrics: {
        'ROC-AUC': syntheticMetrics?.['ROC-AUC'],
        'PR-AUC': syntheticMetrics?.['PR-AUC'],
        Precision: syntheticMetrics?.Precision,
        Recall: syntheticMetrics?.['Recall (Sensitivity)'],
        F1: syntheticMetrics?.['F1 Score'],
      },
    },
    {
      dataset: 'ULB/Worldline real-world credit-card fraud benchmark (held-out test)',
      model: 'ULB MLP · FraudDetectorNet',
      features: '30: Time, V1–V28 (anonymized PCA features), Amount',
      metrics: ulbMetrics.test,
    },
  ];
  const metricNames = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1'];

  return (
    <section className="card animate-in" aria-labelledby="benchmark-records-heading">
      <div className="card-header">
        <span className="card-title" id="benchmark-records-heading">Benchmark Records</span>
      </div>
      <p className="benchmark-note">
        Results are reported independently for each dataset and held-out test split; they are not ranked.
      </p>
      <div className="benchmark-table-wrap">
        <table className="metrics-table benchmark-table">
          <thead>
            <tr>
              <th>Dataset</th><th>Model</th><th>Features</th>
              {metricNames.map((name) => <th key={name}>{name}</th>)}
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr key={record.dataset}>
                <td>{record.dataset}</td>
                <td>{record.model}</td>
                <td className="benchmark-features">{record.features}</td>
                {metricNames.map((name) => (
                  <td key={name}>{format(record.metrics[name])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
