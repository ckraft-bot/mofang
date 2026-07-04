import * as THREE from 'three';

const faceColors = {
  white: 0xffffff,
  yellow: 0xffff00,
  blue: 0x0000ff,
  green: 0x00ff00,
  red: 0xff0000,
  orange: 0xffa500,
  unknown: 0x444444,
};

export function createCubePreview(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 5);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(320, 320);
  container.appendChild(renderer.domElement);

  const cube = new THREE.Group();
  scene.add(cube);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pickableStickers = [];
  let onFaceSelected = () => {};

  const faceDefinitions = [
    { name: 'U', normal: [0, 1, 0], rotation: [Math.PI / 2, 0, 0] },
    { name: 'R', normal: [1, 0, 0], rotation: [0, -Math.PI / 2, 0] },
    { name: 'F', normal: [0, 0, 1], rotation: [0, 0, 0] },
    { name: 'D', normal: [0, -1, 0], rotation: [-Math.PI / 2, 0, 0] },
    { name: 'L', normal: [-1, 0, 0], rotation: [0, Math.PI / 2, 0] },
    { name: 'B', normal: [0, 0, -1], rotation: [0, Math.PI, 0] },
  ];

  const faceGroups = new Map();
  const faceMeshes = new Map();
  faceDefinitions.forEach((face) => {
    const faceGroup = new THREE.Group();
    faceGroup.position.set(face.normal[0] * 0.51, face.normal[1] * 0.51, face.normal[2] * 0.51);
    faceGroup.rotation.set(face.rotation[0], face.rotation[1], face.rotation[2]);
    cube.add(faceGroup);

    const stickers = [];
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        const geometry = new THREE.PlaneGeometry(0.3, 0.3);
        const material = new THREE.MeshBasicMaterial({ color: faceColors.unknown, side: THREE.DoubleSide });
        const sticker = new THREE.Mesh(geometry, material);
        sticker.position.set((col - 1) * 0.34, (1 - row) * 0.34, 0.001);
        sticker.userData.faceName = face.name;
        faceGroup.add(sticker);
        stickers.push(sticker);
        pickableStickers.push(sticker);
      }
    }

    faceGroups.set(face.name, stickers);
    faceMeshes.set(face.name, faceGroup);
  });

  const light = new THREE.AmbientLight(0xffffff, 1.2);
  scene.add(light);

  function handlePointerDown(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects(pickableStickers, true);
    if (intersects.length > 0) {
      const faceName = intersects[0].object.userData.faceName;
      if (faceName) {
        onFaceSelected(faceName);
      }
    }
  }

  renderer.domElement.addEventListener('pointerdown', handlePointerDown);

  function setOnFaceSelected(callback) {
    onFaceSelected = callback;
  }

  function updateFromScan(faceName, colors) {
    const stickers = faceGroups.get(faceName);
    if (!stickers) {
      return;
    }

    stickers.forEach((sticker, index) => {
      const colorName = colors[index] || 'unknown';
      const color = faceColors[colorName] || faceColors.unknown;
      sticker.material.color.set(color);
      sticker.material.opacity = 1;
      sticker.material.transparent = false;
    });
  }

  function setSelectedFace(faceName) {
    faceMeshes.forEach((faceMesh, name) => {
      faceMesh.children.forEach((child) => {
        if (child.material) {
          child.material.color.offsetHSL(0, 0, name === faceName ? 0.08 : 0);
        }
      });
    });
  }

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }

  animate();
  return { cube, renderer, updateFromScan, setSelectedFace, setOnFaceSelected };
}
