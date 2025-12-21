import React, { useMemo, useState } from 'react';
import { useProcessing } from './hooks/useProcessing';
import UploadZone from './components/UploadZone';
import QualitySelector from './components/QualitySelector';
import ProcessingStatus from './components/ProcessingStatus';
import Viewer3D from './components/Viewer3D';
import MetricsPanel from './components/MetricsPanel';
import DownloadPanel from './components/DownloadPanel';
import { Box } from 'lucide-react';

const App = () => {
    const [mode, setMode] = useState('balanced');
    const { status, progress, stage, detail, result, startUpload, reset } = useProcessing();
    const [showMeshPreview, setShowMeshPreview] = useState(true);

    const hasGlb = !!result?.glb;
    const hasMetrics = !!result?.metrics;

    const resultStats = useMemo(() => {
        const time = result?.stats?.time;
        const triangles = result?.stats?.triangles;
        return { time, triangles };
    }, [result]);

    return (
        <div className="app-container">
            {/* Header */}
            <header className="header">
                <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '10px',
                    background: 'rgba(255,255,255,0.05)',
                    padding: '10px 24px',
                    borderRadius: '30px',
                    border: '1px solid rgba(255,255,255,0.1)'
                }}>
                    <Box className="text-primary" size={24} color="var(--primary)" />
                    <span style={{ fontWeight: 700, fontSize: '18px', letterSpacing: '1px' }}>FAST3R FUSION</span>
                </div>
                <h1 className="gradient-text" style={{ fontSize: '48px', fontWeight: 800, marginBottom: '10px' }}>
                    3D Scanner
                </h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '18px', maxWidth: '600px', margin: '0 auto' }}>
                    Turn any object into a production-ready 3D model in seconds.
                </p>
            </header>

            {/* Main Content Area */}
            <main style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>

                {/* State: IDLE - Upload & Config */}
                {status === 'idle' && (
                    <div className="fade-in">
                        <UploadZone onUpload={(files) => startUpload(files, mode)} mode={mode} />
                        <QualitySelector selected={mode} onSelect={setMode} />
                    </div>
                )}

                {/* State: UPLOAD/PROCESSING - Status View */}
                {(status === 'uploading' || status === 'processing') && (
                    <div className="fade-in">
                        <ProcessingStatus
                            status={status}
                            progress={progress}
                            stage={stage}
                            detail={detail}
                        />
                    </div>
                )}

                {/* State: COMPLETE - Results & Viewer */}
                {status === 'complete' && result && (
                    <div className="fade-in">
                        <div className="glass-panel" style={{ padding: '16px 18px', marginTop: '20px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                                <div>
                                    <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>Previews</div>
                                    <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                                        Mesh preview only.
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                                    <button
                                        className="btn-primary"
                                        onClick={() => setShowMeshPreview((v) => !v)}
                                        disabled={!hasGlb}
                                        title={!hasGlb ? 'No mesh output produced' : ''}
                                    >
                                        {showMeshPreview ? 'Hide Mesh' : 'Show Mesh'}
                                    </button>
                                </div>
                            </div>
                            <div style={{ marginTop: 10, color: 'var(--text-muted)', fontSize: 12 }}>
                                {resultStats.time ? <>Time: <b>{resultStats.time}</b> • </> : null}
                                {resultStats.triangles ? <>Triangles: <b>{resultStats.triangles}</b></> : null}
                            </div>
                        </div>

                        {hasGlb && showMeshPreview && <Viewer3D modelUrl={result.glb} />}
                        {hasMetrics && <MetricsPanel metricsUrl={result.metrics} />}

                        <div style={{ textAlign: 'center', marginTop: '20px' }}>
                            <h2 style={{ fontSize: '24px', marginBottom: '8px' }}>Scan Complete!</h2>
                            <p style={{ color: 'var(--text-muted)' }}>
                                Generated {result.stats?.triangles || 'high-quality'} mesh in {result.stats?.time || 'seconds'}
                            </p>
                        </div>

                        <DownloadPanel result={result} onReset={reset} />
                    </div>
                )}

                {/* State: ERROR */}
                {status === 'error' && (
                    <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', borderColor: 'var(--error)' }}>
                        <h3 style={{ color: 'var(--error)', marginBottom: '10px' }}>Processing Failed</h3>
                        <p style={{ marginBottom: '20px' }}>{detail || 'An unexpected error occurred.'}</p>
                        <button className="btn-primary" onClick={reset}>Try Again</button>
                    </div>
                )}

            </main>

            <footer style={{ textAlign: 'center', marginTop: 'auto', color: 'var(--text-muted)', fontSize: '12px' }}>
                <p>Powered by Fast3R Fusion • v1.0.0</p>
            </footer>
        </div>
    );
};

export default App;
