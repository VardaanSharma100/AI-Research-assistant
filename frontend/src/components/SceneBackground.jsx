import { Canvas, useFrame } from "@react-three/fiber";
import { Float, OrbitControls, Stars } from "@react-three/drei";
import { useRef } from "react";

function CoreMesh() {
    const ref = useRef();

    useFrame((_, delta) => {
        if (!ref.current) {
            return;
        }
        ref.current.rotation.x += delta * 0.17;
        ref.current.rotation.y += delta * 0.29;
    });

    return (
        <Float speed={1.4} rotationIntensity={0.35} floatIntensity={0.8}>
            <mesh ref={ref}>
                <icosahedronGeometry args={[1.2, 1]} />
                <meshStandardMaterial color="#6c7980" roughness={0.46} metalness={0.56} />
            </mesh>
        </Float>
    );
}

export default function SceneBackground() {
    return (
        <div className="scene-wrap" aria-hidden="true">
            <Canvas camera={{ position: [0, 0, 4.8], fov: 56 }}>
                <ambientLight intensity={0.34} />
                <pointLight position={[3, 4, 2]} intensity={1.7} color="#bcc9ce" />
                <pointLight position={[-4, -3, -2]} intensity={1.1} color="#5f6c73" />
                <CoreMesh />
                <Stars radius={52} depth={28} count={2600} factor={2.6} saturation={0} fade speed={0.9} />
                <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.25} enablePan={false} />
            </Canvas>
        </div>
    );
}
