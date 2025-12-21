import React from 'react';
import { Download, Share2, RefreshCw } from 'lucide-react';

const DownloadPanel = ({ result, onReset }) => {
    if (!result) return null;

    return (
        <div style={{ marginTop: '30px' }}>
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '20px',
                marginBottom: '40px'
            }}>
                {result.glb && (
                    <a href={result.glb} download className="glass-panel" style={{
                        padding: '20px',
                        textDecoration: 'none',
                        color: 'white',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '15px',
                        transition: 'transform 0.2s'
                    }}>
                        <div style={{ background: '#2563eb', padding: '10px', borderRadius: '8px' }}>
                            <Download size={20} />
                        </div>
                        <div>
                            <div style={{ fontWeight: 600 }}>Download .GLB</div>
                            <div style={{ fontSize: '12px', opacity: 0.7 }}>Textured Mesh</div>
                        </div>
                    </a>
                )}

                {/* Splat downloads intentionally removed from UI (too heavy / browser-hostile). */}

                {result.metrics && (
                    <a href={result.metrics} download className="glass-panel" style={{
                        padding: '20px',
                        textDecoration: 'none',
                        color: 'white',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '15px',
                        transition: 'transform 0.2s'
                    }}>
                        <div style={{ background: '#059669', padding: '10px', borderRadius: '8px' }}>
                            <Download size={20} />
                        </div>
                        <div>
                            <div style={{ fontWeight: 600 }}>Download Metrics</div>
                            <div style={{ fontSize: '12px', opacity: 0.7 }}>metrics.json</div>
                        </div>
                    </a>
                )}
            </div>

            <div style={{ textAlign: 'center' }}>
                <button
                    onClick={onReset}
                    className="glass-panel"
                    style={{
                        background: 'transparent',
                        padding: '12px 30px',
                        color: 'var(--text-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '10px',
                        fontSize: '16px'
                    }}
                >
                    <RefreshCw size={18} /> Scan Another Object
                </button>
            </div>
        </div>
    );
};

export default DownloadPanel;
