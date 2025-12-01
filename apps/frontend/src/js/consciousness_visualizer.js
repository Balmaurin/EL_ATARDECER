/**
 * CONSCIOUSNESS VISUALIZER
 * ========================
 * 3D Particle System representing the AI's consciousness state.
 * Uses Three.js for rendering.
 */

class ConsciousnessVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.particles = null;
        this.targetState = {
            phi: 0.5,
            arousal: 0.5,
            emotion: 'neutral'
        };
        this.currentState = { ...this.targetState };

        this.init();
        this.animate();
    }

    init() {
        // 1. Scene Setup
        this.scene = new THREE.Scene();
        // Fog for depth
        this.scene.fog = new THREE.FogExp2(0x030305, 0.002);

        // 2. Camera
        this.camera = new THREE.PerspectiveCamera(75, this.container.clientWidth / this.container.clientHeight, 0.1, 1000);
        this.camera.position.z = 30;

        // 3. Renderer
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // 4. Particles (The Brain)
        this.createBrainParticles();

        // 5. Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0x00f0ff, 1, 100);
        pointLight.position.set(10, 10, 10);
        this.scene.add(pointLight);

        // 6. Resize Handler
        window.addEventListener('resize', () => this.onResize());
    }

    createBrainParticles() {
        const particleCount = 2000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        const color = new THREE.Color();
        const baseColor = new THREE.Color(0x00f0ff); // Neon Blue

        for (let i = 0; i < particleCount; i++) {
            // Sphere distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);
            const r = 10 + (Math.random() * 2); // Radius with some variation

            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            // Color
            color.set(baseColor);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            // Size
            sizes[i] = Math.random() * 0.5;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        // Shader Material for glowing particles
        const material = new THREE.PointsMaterial({
            size: 0.5,
            vertexColors: true,
            map: this.createCircleTexture(),
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }

    createCircleTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const context = canvas.getContext('2d');
        const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
        gradient.addColorStop(0, 'rgba(255,255,255,1)');
        gradient.addColorStop(0.2, 'rgba(255,255,255,0.8)');
        gradient.addColorStop(0.5, 'rgba(255,255,255,0.2)');
        gradient.addColorStop(1, 'rgba(0,0,0,0)');
        context.fillStyle = gradient;
        context.fillRect(0, 0, 32, 32);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    updateState(data) {
        if (!data) return;

        this.targetState = {
            phi: data.phi_value || 0.5,
            arousal: data.arousal || 0.5,
            emotion: data.emotion || 'neutral'
        };

        // Update color based on emotion
        const colors = {
            joy: 0xffd700,      // Gold
            sadness: 0x00bfff,  // Deep Sky Blue
            anger: 0xff4500,    // Orange Red
            fear: 0x800080,     // Purple
            trust: 0x00fa9a,    // Medium Spring Green
            neutral: 0x00f0ff,  // Cyan (Default)
            love: 0xff69b4,     // Hot Pink
            optimism: 0xffa500  // Orange
        };

        const targetColorHex = colors[this.targetState.emotion.toLowerCase()] || colors.neutral;
        const targetColor = new THREE.Color(targetColorHex);

        // Tween particle colors
        const geometry = this.particles.geometry;
        const particleColors = geometry.attributes.color.array;

        for (let i = 0; i < particleColors.length; i += 3) {
            // Simple lerp for demo (in production use proper tweening)
            particleColors[i] = targetColor.r;
            particleColors[i + 1] = targetColor.g;
            particleColors[i + 2] = targetColor.b;
        }
        geometry.attributes.color.needsUpdate = true;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.particles) {
            // Rotation based on arousal
            const rotationSpeed = 0.001 + (this.targetState.arousal * 0.005);
            this.particles.rotation.y += rotationSpeed;
            this.particles.rotation.z += rotationSpeed * 0.5;

            // Pulse effect based on Phi
            const time = Date.now() * 0.001;
            const scale = 1 + Math.sin(time * (this.targetState.phi * 5)) * 0.05;
            this.particles.scale.set(scale, scale, scale);
        }

        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        if (!this.container || !this.camera || !this.renderer) return;

        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

// Export to global scope
window.ConsciousnessVisualizer = ConsciousnessVisualizer;
