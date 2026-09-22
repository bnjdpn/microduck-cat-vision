import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

export const FEEDER_ARENA_TARGETS = Object.freeze({
  left: Object.freeze({ x: 0.88, z: -0.58, assignedTo: "Nooby" }),
  right: Object.freeze({ x: 0.88, z: 0.58, assignedTo: "Jaina" }),
});

const SURFACE_MAP_SIZE = 1024;
const roundedGeometryCache = new Map();

function seededNoise(x, y, seed) {
  let value = Math.imul(x + seed * 17, 374761393) ^ Math.imul(y + seed * 31, 668265263);
  value = Math.imul(value ^ (value >>> 13), 1274126177);
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295;
}

function makeSurfaceTexture(kind, base, seed, anisotropic = false) {
  const canvas = document.createElement("canvas");
  canvas.width = SURFACE_MAP_SIZE;
  canvas.height = SURFACE_MAP_SIZE;
  const context = canvas.getContext("2d", { alpha: false });
  const image = context.createImageData(SURFACE_MAP_SIZE, SURFACE_MAP_SIZE);
  const [red, green, blue] = base;
  for (let y = 0; y < SURFACE_MAP_SIZE; y += 1) {
    for (let x = 0; x < SURFACE_MAP_SIZE; x += 1) {
      const line = anisotropic ? Math.sin(y * 0.29) * 0.5 + 0.5 : 0;
      const grain = seededNoise(x, anisotropic ? Math.floor(y / 2) : y, seed);
      const broad = seededNoise(Math.floor(x / 23), Math.floor(y / 23), seed + 11);
      const value = kind === "albedo"
        ? (grain - 0.5) * 7 + (broad - 0.5) * 4
        : kind === "roughness"
          ? 118 + grain * 76 + line * 18
          : kind === "ao"
            ? 228 + broad * 25
            : 112 + grain * 42 + line * 22;
      const offset = (y * SURFACE_MAP_SIZE + x) * 4;
      image.data[offset] = kind === "albedo" ? Math.max(0, Math.min(255, red + value)) : value;
      image.data[offset + 1] = kind === "albedo" ? Math.max(0, Math.min(255, green + value)) : value;
      image.data[offset + 2] = kind === "albedo" ? Math.max(0, Math.min(255, blue + value)) : value;
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(anisotropic ? 2 : 1.5, anisotropic ? 10 : 1.5);
  texture.colorSpace = kind === "albedo" ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function makeSurfaceSet(base, seed, anisotropic = false) {
  return {
    map: makeSurfaceTexture("albedo", base, seed, anisotropic),
    roughnessMap: makeSurfaceTexture("roughness", [0, 0, 0], seed + 1, anisotropic),
    bumpMap: makeSurfaceTexture("bump", [0, 0, 0], seed + 2, anisotropic),
    aoMap: makeSurfaceTexture("ao", [0, 0, 0], seed + 3, anisotropic),
  };
}

function ensureUv2(geometry) {
  const uv = geometry.getAttribute("uv");
  if (uv && !geometry.getAttribute("uv2")) geometry.setAttribute("uv2", uv.clone());
  return geometry;
}

function makeLabelSprite(text, accent, { width = 0.36, height = 0.085 } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = 768;
  canvas.height = 180;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(9, 10, 13, 0.94)";
  context.strokeStyle = `#${accent.toString(16).padStart(6, "0")}`;
  context.lineWidth = 12;
  context.beginPath();
  context.roundRect(10, 10, canvas.width - 20, canvas.height - 20, 24);
  context.fill();
  context.stroke();
  context.fillStyle = "#fffaf0";
  context.font = "700 68px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2 + 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width, height, 1);
  sprite.renderOrder = 5;
  return sprite;
}

function makeCatMarker(label) {
  const isNooby = label === "Nooby";
  const root = new THREE.Group();
  root.name = `cat-marker-${label.toLowerCase()}`;

  const coat = new THREE.MeshStandardMaterial({
    color: isNooby ? 0x2a211d : 0xc7c9c8,
    roughness: 0.96,
    metalness: 0,
  });
  const secondary = new THREE.MeshStandardMaterial({
    color: isNooby ? 0x9b552d : 0xf4f1e8,
    roughness: 0.95,
    metalness: 0,
  });
  const darkStripe = new THREE.MeshStandardMaterial({
    color: isNooby ? 0x0e1012 : 0x686b6f,
    roughness: 0.98,
    metalness: 0,
  });

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 16), coat);
  body.name = `${root.name}-body`;
  body.scale.set(1.2, 1.25, 0.86);
  body.position.set(-0.015, 0.145, 0);
  body.castShadow = true;
  root.add(body);

  const haunchGeometry = new THREE.SphereGeometry(0.055, 18, 12);
  for (const [index, z] of [-0.062, 0.062].entries()) {
    const haunch = new THREE.Mesh(haunchGeometry, index === 0 && isNooby ? secondary : coat);
    haunch.name = `${root.name}-haunch-${index}`;
    haunch.scale.set(1.15, 0.8, 0.9);
    haunch.position.set(-0.085, 0.062, z);
    haunch.castShadow = true;
    root.add(haunch);
  }

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.075, 22, 14), isNooby ? coat : secondary);
  head.name = `${root.name}-head`;
  head.scale.set(0.92, 0.9, 1);
  head.position.set(0.105, 0.205, 0);
  head.castShadow = true;
  root.add(head);

  const earGeometry = new THREE.ConeGeometry(0.032, 0.075, 4);
  for (const [index, z] of [-0.044, 0.044].entries()) {
    const ear = new THREE.Mesh(earGeometry, index === 0 && isNooby ? secondary : coat);
    ear.name = `${root.name}-ear-${index}`;
    ear.position.set(0.105, 0.285, z);
    ear.rotation.y = Math.PI / 4;
    ear.castShadow = true;
    root.add(ear);
  }

  const patch = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 10), isNooby ? secondary : darkStripe);
  patch.name = `${root.name}-back-patch`;
  patch.scale.set(1.4, 0.2, 0.65);
  patch.position.set(-0.01, 0.255, isNooby ? -0.035 : 0.018);
  root.add(patch);

  const tailCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.11, 0.12, 0.05),
    new THREE.Vector3(-0.19, 0.13, 0.10),
    new THREE.Vector3(-0.22, 0.22, 0.13),
    new THREE.Vector3(-0.17, 0.29, 0.14),
  ]);
  const tail = new THREE.Mesh(new THREE.TubeGeometry(tailCurve, 18, 0.018, 8, false), coat);
  tail.name = `${root.name}-tail`;
  tail.castShadow = true;
  root.add(tail);

  const nameplate = makeLabelSprite(label.toUpperCase(), isNooby ? 0xf29a52 : 0x8fd2ff, {
    width: 0.24,
    height: 0.06,
  });
  nameplate.name = `${root.name}-name`;
  nameplate.position.set(-0.03, 0.40, 0);
  root.add(nameplate);

  const haloMaterial = new THREE.MeshBasicMaterial({
    color: 0xff7a2f,
    transparent: true,
    opacity: 0.65,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const halo = new THREE.Mesh(new THREE.RingGeometry(0.145, 0.17, 40), haloMaterial);
  halo.name = `${root.name}-alert-halo`;
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.008;
  root.add(halo);

  root.userData.tail = tail;
  root.userData.halo = halo;
  root.visible = false;
  return root;
}

function roundedBox(width, height, depth, radius, material, name) {
  const key = `${width}:${height}:${depth}:${radius}`;
  let geometry = roundedGeometryCache.get(key);
  if (!geometry) {
    geometry = ensureUv2(new RoundedBoxGeometry(width, height, depth, 4, radius));
    roundedGeometryCache.set(key, geometry);
  }
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function makeBlockoutMaterials() {
  const rubber = makeSurfaceSet([74, 78, 86], 4);
  const plastic = makeSurfaceSet([243, 240, 231], 9);
  const brushedSteel = makeSurfaceSet([174, 180, 186], 14, true);
  return {
    mat: new THREE.MeshStandardMaterial({
      ...rubber,
      color: 0xffffff,
      roughness: 0.94,
      metalness: 0,
      bumpScale: 0.008,
      aoMapIntensity: 0.65,
    }),
    white: new THREE.MeshPhysicalMaterial({
      ...plastic,
      color: 0xffffff,
      roughness: 0.56,
      metalness: 0,
      clearcoat: 0.12,
      clearcoatRoughness: 0.72,
      bumpScale: 0.002,
      aoMapIntensity: 0.45,
    }),
    dark: new THREE.MeshStandardMaterial({ color: 0x12151b, roughness: 0.4, metalness: 0.02 }),
    steel: new THREE.MeshStandardMaterial({
      ...brushedSteel,
      color: 0xffffff,
      roughness: 0.29,
      metalness: 0.84,
      bumpScale: 0.003,
      aoMapIntensity: 0.4,
    }),
    water: new THREE.MeshPhysicalMaterial({
      color: 0x86d7ec,
      roughness: 0.12,
      metalness: 0,
      transmission: 0.22,
      transparent: true,
      opacity: 0.78,
    }),
  };
}

/**
 * Blockout pass for the feeding corner. The hierarchy and target map are
 * intentionally stable so structural/detail passes can refine meshes without
 * changing the runtime API used by the cat-vision reaction.
 */
export function createFeedingCornerScene() {
  const group = new THREE.Group();
  group.name = "feeder-scene";
  const materials = makeBlockoutMaterials();
  const nodes = {};
  const meshes = {};
  const sockets = {};
  const markers = {};
  const pelletGeometry = new THREE.DodecahedronGeometry(0.012, 0);
  const pelletMaterial = new THREE.MeshStandardMaterial({
    color: 0x6f3c21,
    roughness: 0.96,
    metalness: 0,
  });
  const pelletInstances = new THREE.InstancedMesh(pelletGeometry, pelletMaterial, 14);
  pelletInstances.name = "xiaomi-food-pellets";
  pelletInstances.castShadow = true;
  const pelletTransform = new THREE.Object3D();
  let pelletInstanceIndex = 0;
  const statusMaterials = [
    new THREE.MeshStandardMaterial({
      color: 0xffb15c,
      emissive: 0x7a2d05,
      emissiveIntensity: 0.9,
      roughness: 0.3,
    }),
    new THREE.MeshStandardMaterial({
      color: 0xb7c5cb,
      emissive: 0x111619,
      emissiveIntensity: 0.9,
      roughness: 0.3,
    }),
  ];
  const statusGeometry = new THREE.SphereGeometry(0.008, 12, 8);
  group.add(pelletInstances);
  meshes[pelletInstances.name] = pelletInstances;

  const mat = new THREE.Mesh(ensureUv2(new THREE.PlaneGeometry(0.82, 1.72)), materials.mat);
  mat.name = "feeder-mat";
  mat.rotation.x = -Math.PI / 2;
  mat.position.set(1.085, 0.002, 0);
  mat.receiveShadow = true;
  group.add(mat);
  meshes.mat = mat;

  // A restrained appliance light rig complements the arena environment:
  // warm key from camera-left, cool fill from camera-right and a low rim
  // behind the fountain. It keeps white plastic and steel separable without
  // changing the simulator renderer/tone-mapping contract.
  const lightTarget = new THREE.Object3D();
  lightTarget.position.set(1.22, 0.18, 0);
  group.add(lightTarget);

  const keyLight = new THREE.SpotLight(0xffdfbd, 2.0, 4.0, 0.72, 0.65, 1.5);
  keyLight.name = "feeder-key-light";
  keyLight.position.set(0.15, 1.35, -1.1);
  keyLight.target = lightTarget;
  group.add(keyLight);

  const fillLight = new THREE.PointLight(0xb9d9ff, 0.55, 2.8, 1.7);
  fillLight.name = "feeder-fill-light";
  fillLight.position.set(0.55, 0.78, 1.15);
  group.add(fillLight);

  const rimLight = new THREE.PointLight(0xffa65c, 0.32, 1.8, 1.8);
  rimLight.name = "feeder-rim-light";
  rimLight.position.set(1.58, 0.78, 0);
  group.add(rimLight);

  const addApplianceRoot = (id, position) => {
    const root = new THREE.Group();
    root.name = id;
    root.position.copy(position);
    group.add(root);
    nodes[id] = root;
    return root;
  };

  for (const [side, z] of [["left", -0.58], ["right", 0.58]]) {
    const root = addApplianceRoot(`xiaomi-${side}`, new THREE.Vector3(1.25, 0, z));
    const base = roundedBox(0.30, 0.21, 0.31, 0.045, materials.white, `xiaomi-${side}-base`);
    base.position.set(0, 0.125, 0);
    root.add(base);
    meshes[`xiaomi-${side}-base`] = base;

    const hopper = roundedBox(0.285, 0.34, 0.295, 0.05, materials.white, `xiaomi-${side}-hopper`);
    hopper.position.set(0.008, 0.365, 0);
    root.add(hopper);
    meshes[`xiaomi-${side}-hopper`] = hopper;

    const dispenserMouth = roundedBox(0.018, 0.075, 0.145, 0.007, materials.dark, `xiaomi-${side}-mouth`);
    dispenserMouth.position.set(-0.155, 0.13, 0);
    root.add(dispenserMouth);
    meshes[`xiaomi-${side}-mouth`] = dispenserMouth;

    const lid = roundedBox(0.296, 0.026, 0.306, 0.012, materials.white, `xiaomi-${side}-lid`);
    lid.position.set(0.008, 0.545, 0);
    root.add(lid);
    meshes[`xiaomi-${side}-lid`] = lid;

    const lidSeam = roundedBox(0.29, 0.009, 0.30, 0.01, materials.dark, `xiaomi-${side}-lid-seam`);
    lidSeam.position.set(0.008, 0.528, 0);
    root.add(lidSeam);
    meshes[`xiaomi-${side}-lid-seam`] = lidSeam;

    for (let index = 0; index < 3; index += 1) {
      const led = new THREE.Mesh(
        statusGeometry,
        statusMaterials[index === 0 ? 0 : 1],
      );
      led.name = `xiaomi-${side}-status-${index}`;
      led.position.set(-0.139, 0.27, -0.028 + index * 0.028);
      root.add(led);
      meshes[led.name] = led;
    }

    // Both real feeders face the camera/duck. The tray projects from the
    // appliance root toward -X, so its centre also supplies a stable socket.
    const tray = roundedBox(0.205, 0.045, 0.245, 0.018, materials.white, `xiaomi-${side}-tray`);
    tray.position.set(-0.205, 0.038, 0);
    root.add(tray);
    meshes[`xiaomi-${side}-tray`] = tray;

    const bowl = roundedBox(0.155, 0.017, 0.205, 0.014, materials.steel, `xiaomi-${side}-bowl`);
    bowl.position.set(-0.225, 0.066, 0);
    root.add(bowl);
    meshes[`xiaomi-${side}-bowl`] = bowl;

    for (let index = 0; index < 7; index += 1) {
      pelletTransform.position.set(
        root.position.x - 0.255 + (index % 3) * 0.032,
        0.081 + (index % 2) * 0.004,
        root.position.z - 0.055 + Math.floor(index / 3) * 0.052,
      );
      pelletTransform.rotation.set(index * 0.31, index * 0.67, index * 0.19);
      pelletTransform.updateMatrix();
      pelletInstances.setMatrixAt(pelletInstanceIndex, pelletTransform.matrix);
      pelletInstanceIndex += 1;
    }

    const target = FEEDER_ARENA_TARGETS[side];
    const socket = new THREE.Object3D();
    socket.name = `feeding-${side}`;
    socket.position.set(target.x - root.position.x, 0, target.z - root.position.z);
    root.add(socket);
    sockets[`feeding-${side}`] = socket;

    const assignment = makeLabelSprite(
      side === "left" ? "NOOBY · GAUCHE" : "JAINA · DROITE",
      side === "left" ? 0xf29a52 : 0x8fd2ff,
      { width: 0.28, height: 0.06 },
    );
    assignment.name = `assignment-${side}`;
    assignment.position.set(-0.19, 0.115, 0);
    root.add(assignment);
    meshes[assignment.name] = assignment;
  }
  pelletInstances.instanceMatrix.needsUpdate = true;

  const fountainRoot = addApplianceRoot("petlibro-centre", new THREE.Vector3(1.25, 0, 0));
  const fountain = roundedBox(0.36, 0.18, 0.48, 0.055, materials.dark, "petlibro-body");
  fountain.position.y = 0.10;
  fountainRoot.add(fountain);
  meshes["petlibro-body"] = fountain;

  const rim = roundedBox(0.345, 0.024, 0.465, 0.04, materials.steel, "petlibro-rim");
  rim.position.y = 0.196;
  fountainRoot.add(rim);
  meshes["petlibro-rim"] = rim;

  const water = roundedBox(0.31, 0.011, 0.43, 0.035, materials.water, "petlibro-water");
  water.position.y = 0.212;
  fountainRoot.add(water);
  meshes["petlibro-water"] = water;

  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.027, 0.034, 0.15, 20), materials.dark);
  stem.name = "petlibro-spout-stem";
  stem.position.set(0.045, 0.285, 0);
  stem.castShadow = true;
  fountainRoot.add(stem);
  meshes[stem.name] = stem;

  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.068, 0.018, 24), materials.dark);
  cap.name = "petlibro-spout-cap";
  cap.position.set(0.045, 0.368, 0);
  cap.castShadow = true;
  fountainRoot.add(cap);
  meshes[cap.name] = cap;

  const fallingWater = new THREE.Mesh(
    new THREE.CylinderGeometry(0.012, 0.018, 0.145, 12),
    materials.water,
  );
  fallingWater.name = "petlibro-water-column";
  fallingWater.position.set(0.045, 0.285, 0);
  fountainRoot.add(fallingWater);
  meshes[fallingWater.name] = fallingWater;

  for (let index = 0; index < 3; index += 1) {
    const ripple = new THREE.Mesh(
      new THREE.TorusGeometry(0.045 + index * 0.034, 0.003, 6, 32),
      materials.water,
    );
    ripple.name = `petlibro-ripple-${index}`;
    ripple.rotation.x = Math.PI / 2;
    ripple.position.set(0.045, 0.221 + index * 0.001, 0);
    fountainRoot.add(ripple);
    meshes[ripple.name] = ripple;
  }

  for (const label of ["Nooby", "Jaina"]) {
    const marker = makeCatMarker(label);
    marker.position.set(FEEDER_ARENA_TARGETS.left.x, 0, FEEDER_ARENA_TARGETS.left.z);
    group.add(marker);
    markers[label] = marker;
    nodes[marker.name] = marker;
  }

  function setViolations(violations = []) {
    for (const marker of Object.values(markers)) marker.visible = false;
    for (const violation of violations) {
      const marker = markers[violation?.label];
      const target = FEEDER_ARENA_TARGETS[violation?.side];
      if (!marker || !target) continue;
      marker.position.set(target.x, 0, target.z);
      marker.userData.side = violation.side;
      marker.visible = true;
    }
    publishStats();
  }

  function update(nowMs = performance.now()) {
    const phase = nowMs * 0.0045;
    for (const [index, marker] of Object.values(markers).entries()) {
      if (!marker.visible) continue;
      marker.position.y = 0.006 + Math.sin(phase + index) * 0.004;
      marker.userData.tail.rotation.x = Math.sin(phase * 0.75 + index) * 0.09;
      marker.userData.halo.material.opacity = 0.42 + (Math.sin(phase * 1.4) * 0.5 + 0.5) * 0.4;
    }
  }

  function getStats() {
    const geometries = new Set();
    const materialsSeen = new Set();
    const textures = new Set();
    let triangles = 0;
    let drawCalls = 0;
    let visibleObjects = 0;
    const visibleInHierarchy = (object) => {
      for (let current = object; current; current = current.parent) {
        if (!current.visible) return false;
      }
      return true;
    };
    group.traverse((object) => {
      if ((!object.isMesh && !object.isSprite) || !visibleInHierarchy(object)) return;
      visibleObjects += 1;
      drawCalls += Array.isArray(object.material) ? object.material.length : 1;
      if (object.isSprite) {
        triangles += 2;
      } else if (object.geometry) {
        geometries.add(object.geometry.uuid);
        const baseTriangles = object.geometry.index
          ? object.geometry.index.count / 3
          : object.geometry.getAttribute("position").count / 3;
        triangles += baseTriangles * (object.isInstancedMesh ? object.count : 1);
      }
      for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
        if (!material) continue;
        materialsSeen.add(material.uuid);
        for (const value of Object.values(material)) if (value?.isTexture) textures.add(value.uuid);
      }
    });
    return {
      triangles: Math.round(triangles),
      drawCalls,
      visibleObjects,
      uniqueGeometries: geometries.size,
      uniqueMaterials: materialsSeen.size,
      uniqueTextures: textures.size,
      fpsTarget: 60,
      instancedPellets: pelletInstances.count,
    };
  }

  function publishStats() {
    group.userData.stats = getStats();
    document.documentElement.dataset.feederSceneStats = JSON.stringify(group.userData.stats);
  }

  group.userData.sculptRuntime = {
    nodes,
    meshes,
    sockets,
    colliders: {
      "xiaomi-left": { type: "box", size: [0.50, 0.56, 0.31] },
      "petlibro-centre": { type: "box", size: [0.36, 0.38, 0.48] },
      "xiaomi-right": { type: "box", size: [0.50, 0.56, 0.31] },
    },
    destructionGroups: {},
    markers,
  };
  publishStats();

  return { group, nodes, meshes, sockets, markers, materials, setViolations, update, getStats };
}
