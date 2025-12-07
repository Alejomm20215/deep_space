import React from 'react';
import { Loader2, CheckCircle2 } from 'lucide-react';

const ProcessingStatus = ({ status, progress, stage, detail }) => {
    return (
        <div className="glass-panel" style={{ padding: '40px', marginTop: '40px', textAlign: 'center' }}>
            <div style={{ marginBottom: '20px' }}>
                {status === 'complete' ? (
                    <CheckCircle2 size={48} color="var(--success)" style={{ margin: '0 auto' }} />
                ) : (
                    <Loader2 size={48} className="spin" color="var(--primary)" style={{ margin: '0 auto', animation: 'spin 2s linear infinite' }} />
                )}
            </div>

            <h3 style={{ fontSize: '24px', marginBottom: '10px' }}>
                {status === 'complete' ? 'Processing Complete!' : 'Processing Scan...'}
            </h3>

            <p style={{ color: 'var(--text-muted)', marginBottom: '30px' }}>
                {detail || 'Initializing...'}
            </p>

            {/* Progress Bar */}
            <div style={{
                width: '100%',
                height: '6px',
                background: 'rgba(255,255,255,0.1)',
                borderRadius: '3px',
                overflow: 'hidden',
                position: 'relative'
            }}>
                <div style={{
                    width: `${progress}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, var(--primary), var(--accent))',
                    transition: 'width 0.5s ease',
                    boxShadow: '0 0 10px var(--primary)'
                }} />
            </div>

            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: '10px',
                fontSize: '14px',
                color: 'var(--text-muted)'
            }}>
                <span>{stage}</span>
                <span>{Math.round(progress)}%</span>
            </div>

            <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    );
};

export default ProcessingStatus;
