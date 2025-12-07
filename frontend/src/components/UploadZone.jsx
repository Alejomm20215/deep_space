import React, { useRef, useState } from 'react';
import { Upload, FileVideo, Image, Camera, AlertCircle } from 'lucide-react';
import '../styles/app.css';

const UploadZone = ({ onUpload, mode }) => {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setIsDragging(true);
        } else if (e.type === 'dragleave') {
            setIsDragging(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            onUpload(e.dataTransfer.files);
        }
    };

    // Determine requirements based on mode
    const getInstructions = () => {
        if (mode === 'fastest') {
            return (
                <div style={{ marginTop: '20px', padding: '15px', background: 'rgba(251, 191, 36, 0.1)', borderRadius: '12px', border: '1px solid rgba(251, 191, 36, 0.2)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', color: '#fbbf24', fontWeight: 600 }}>
                        <AlertCircle size={18} />
                        <span>Fastest Mode Requirement</span>
                    </div>
                    <p style={{ fontSize: '14px', lineHeight: '1.4', margin: 0, color: '#d1d5db' }}>
                        Upload exactly <strong>4 overlapping photos</strong> taken from different angles around the object.
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div>
            {/* Instructions Panel */}
            <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '18px', marginBottom: '16px' }}>
                    <Camera size={20} color="var(--primary)" />
                    <span>How to Capture</span>
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                    <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
                        <div style={{ fontWeight: 600, marginBottom: '4px', color: '#fff' }}>1. Center Object</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Keep the object in the middle of every frame. Avoid moving the object itself.</div>
                    </div>
                    <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
                        <div style={{ fontWeight: 600, marginBottom: '4px', color: '#fff' }}>2. 360° Coverage</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Move around the object. Ensure adjacent photos have 50%+ overlap.</div>
                    </div>
                    <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
                        <div style={{ fontWeight: 600, marginBottom: '4px', color: '#fff' }}>3. Lighting</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Ensure even lighting. Avoid extreme shadows or transparent/reflective surfaces.</div>
                    </div>
                </div>

                {getInstructions()}
            </div>

            <div
                className={`glass-panel upload-zone ${isDragging ? 'dragging' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                    border: isDragging ? '2px dashed var(--primary)' : '2px dashed rgba(255,255,255,0.1)',
                    padding: '60px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                }}
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    multiple // Allow multiple files
                    accept="video/*,image/*"
                    onChange={(e) => e.target.files && e.target.files.length > 0 && onUpload(e.target.files)}
                />

                <div style={{
                    background: 'rgba(99, 102, 241, 0.1)',
                    borderRadius: '50%',
                    width: '80px',
                    height: '80px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 20px auto'
                }}>
                    <Upload size={40} color="var(--primary)" />
                </div>

                <h3 style={{ fontSize: '24px', marginBottom: '10px' }}>
                    Drop {mode === 'fastest' ? '4 photos' : 'video or photos'} here
                </h3>
                <p style={{ color: 'var(--text-muted)' }}>or click to browse local files</p>

                <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '30px', opacity: 0.6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Image size={16} /> <span>JPG, PNG ({mode === 'fastest' ? '4 inputs' : 'Sequence'})</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FileVideo size={16} /> <span>Video (MP4, MOV)</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UploadZone;
