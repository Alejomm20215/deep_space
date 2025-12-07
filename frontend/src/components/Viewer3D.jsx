import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stage, useGLTF } from '@react-three/drei';

const Model = ({ url }) => {
    const { scene } = useGLTF(url);
    return <primitive object={scene} />;
};

const Viewer3D = ({ modelUrl }) => {
    if (!modelUrl) return null;

    return (
        <div className="glass-panel" style={{
            height: '500px',
            width: '100%',
            marginTop: '30px',
            position: 'relative',
            overflow: 'hidden'
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
                backdropFilter: 'blur(4px)'
            }}>
                Interactive 3D Preview
            </div>

            <Canvas shadows dpr={[1, 2]} camera={{ fov: 50 }}>
                <Suspense fallback={null}>
                    <Stage environment="city" intensity={0.6}>
                        <Model url={modelUrl} />
                    </Stage>
                </Suspense>
                <OrbitControls autoRotate />
            </Canvas>

            <div style={{
                position: 'absolute',
                bottom: '20px',
                left: '0',
                width: '100%',
                textAlign: 'center',
                opacity: 0.5,
                fontSize: '12px',
                pointerEvents: 'none'
            }}>
                Drag to rotate • Scroll to zoom
            </div>
        </div>
    );
};

export default Viewer3D;
