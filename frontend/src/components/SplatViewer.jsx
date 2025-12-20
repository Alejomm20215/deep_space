import React, { useEffect, useRef } from 'react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const SplatViewer = ({ splatUrl }) => {
    const containerRef = useRef(null);
    const viewerRef = useRef(null);

    useEffect(() => {
        if (!containerRef.current || !splatUrl) return;

        // Clean up any previous viewer instance
        if (viewerRef.current) {
            try {
                viewerRef.current.dispose?.();
            } catch {
                // ignore
            }
            viewerRef.current = null;
            containerRef.current.innerHTML = '';
        }

        const viewer = new GaussianSplats3D.Viewer({
            rootElement: containerRef.current,
            sharedMemoryForWorkers: false,
            cameraUp: [0, 1, 0],
            dynamicScene: false,
        });
        viewerRef.current = viewer;

        viewer
            .addSplatScene(splatUrl, {
                showLoadingUI: true,
                progressiveLoad: true,
                position: [0, 0, 0],
                rotation: [0, 0, 0, 1],
                scale: [1, 1, 1],
            })
            .then(() => viewer.start())
            .catch((e) => {
                // eslint-disable-next-line no-console
                console.error('Failed to load splat:', e);
            });

        return () => {
            try {
                viewer.dispose?.();
            } catch {
                // ignore
            }
        };
    }, [splatUrl]);

    if (!splatUrl) return null;

    return (
        <div className="glass-panel" style={{
            height: '500px',
            width: '100%',
            marginTop: '30px',
            position: 'relative',
            overflow: 'hidden',
        }}>
            <div style={{
                position: 'absolute',
                top: '20px',
                left: '20px',
                zIndex: 10,
                background: 'rgba(0,0,0,0.5)',
                padding: '8px 16px',
                borderRadius: '20px',
                fontSize: '12px',
                backdropFilter: 'blur(4px)',
            }}>
                Gaussian Splat Preview
            </div>

            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
        </div>
    );
};

export default SplatViewer;

