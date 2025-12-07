import React from 'react';
import { Zap, Scale, Sparkles } from 'lucide-react';

const QualitySelector = ({ selected, onSelect }) => {
    const modes = [
        {
            id: 'fastest',
            icon: <Zap size={20} />,
            label: 'Fastest',
            time: '~20 sec',
            desc: 'Quick preview, max speed',
            color: '#fbbf24' // amber-400
        },
        {
            id: 'balanced',
            icon: <Scale size={20} />,
            label: 'Balanced',
            time: '~60 sec',
            desc: 'Good trade-off',
            color: '#6366f1' // indigo-500
        },
        {
            id: 'quality',
            icon: <Sparkles size={20} />,
            label: 'Max Quality',
            time: '~3 min',
            desc: 'Best geometry & texture',
            color: '#ec4899' // pink-500
        }
    ];

    return (
        <div style={{ marginTop: '30px' }}>
            <p style={{ color: 'var(--text-muted)', marginBottom: '16px', textAlign: 'center' }}>Select Processing Mode</p>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '20px'
            }}>
                {modes.map((mode) => (
                    <div
                        key={mode.id}
                        className="glass-panel"
                        onClick={() => onSelect(mode.id)}
                        style={{
                            padding: '20px',
                            cursor: 'pointer',
                            border: selected === mode.id ? `1px solid ${mode.color}` : '1px solid transparent',
                            background: selected === mode.id ? `rgba(${parseInt(mode.color.slice(1, 3), 16)}, ${parseInt(mode.color.slice(3, 5), 16)}, ${parseInt(mode.color.slice(5, 7), 16)}, 0.1)` : 'var(--glass-bg)',
                            transition: 'all 0.2s ease',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            textAlign: 'center'
                        }}
                    >
                        <div style={{ color: mode.color, marginBottom: '10px' }}>
                            {mode.icon}
                        </div>
                        <div style={{ fontWeight: '600', marginBottom: '4px' }}>{mode.label}</div>
                        <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '8px' }}>{mode.time}</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{mode.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default QualitySelector;
