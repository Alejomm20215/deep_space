import React, { useEffect, useState } from 'react';

const MetricsPanel = ({ metricsUrl }) => {
    const [metrics, setMetrics] = useState(null);
    const [err, setErr] = useState(null);

    useEffect(() => {
        if (!metricsUrl) return;
        let cancelled = false;

        fetch(metricsUrl)
            .then((r) => {
                if (!r.ok) throw new Error(`Failed to load metrics (${r.status})`);
                return r.json();
            })
            .then((json) => {
                if (!cancelled) setMetrics(json);
            })
            .catch((e) => {
                if (!cancelled) setErr(e.message);
            });

        return () => { cancelled = true; };
    }, [metricsUrl]);

    if (!metricsUrl) return null;

    return (
        <div className="glass-panel" style={{ padding: '20px', marginTop: '20px' }}>
            <div style={{ fontWeight: 700, marginBottom: '10px' }}>Run Metrics</div>
            {err && <div style={{ opacity: 0.8 }}>Failed to load metrics: {err}</div>}
            {!err && !metrics && <div style={{ opacity: 0.8 }}>Loading metrics…</div>}
            {metrics && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
                    <div><b>Frames:</b> {metrics.frames}</div>
                    <div><b>Mesh faces:</b> {metrics.mesh_faces ?? 'n/a'}</div>
                    <div><b>Elapsed:</b> {metrics.elapsed_seconds ? `${metrics.elapsed_seconds.toFixed(1)}s` : 'n/a'}</div>
                    <div><b>GLB size:</b> {metrics.sizes_bytes?.glb ?? 0} bytes</div>
                </div>
            )}
        </div>
    );
};

export default MetricsPanel;

