import * as THREE from "three";
import { FBXLoader } from "jsm/loaders/FBXLoader.js";
import { GLTFLoader } from "jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "jsm/controls/OrbitControls.js";
import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
import getLayer from "./libs/getLayer.js";

const w = window.innerWidth;
const h = window.innerHeight;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b12);

const camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
camera.position.set(0, 1.4, 5.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(w, h);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

const manager = new THREE.LoadingManager();
const loader = new FBXLoader(manager);
const textureLoader = new THREE.TextureLoader();
const path = "./assets/Y Bot.fbx";

let character;
const sceneData = {
  character: null,
  animations: [],
};

const statusEl = document.getElementById("status");
const videoInput = document.getElementById("videoInput");
const sourceVideo = document.getElementById("sourceVideo");
const videoPanel = document.getElementById("videoPanel");

loader.load(path, (fbx) => {
  function getMaterial() {
    const material = new THREE.MeshMatcapMaterial({
      matcap: textureLoader.load("./assets/fire-edge-blue.jpg"),
    });
    return material;
  }

  function initCharacter(fbx) {
    const char = fbx;
    char.scale.setScalar(0.02);
    char.position.set(0, -1.5, 0);

    char.traverse((c) => {
      if (c.isMesh) {
        if (c.material.name === "Alpha_Body_MAT") {
          c.material = getMaterial();
        }
        c.castShadow = true;
      }
    });

    const mixer = new THREE.AnimationMixer(char);
    const update = (t) => {
      mixer.update(0.02);
    };
    char.userData = { mixer, update };
    return char;
  }

  character = initCharacter(fbx);
  sceneData.character = character;
});

const animations = [
  "Being Electrocuted",
  "Drunk Walk",
  "Floating",
  "Idle",
  "Joyful Jump",
  "Kneeling Pointing",
  "Low Crawl",
  "Male Dance Pose",
  "Neutral Idle",
  "Reaction",
  "Spat In Face",
  "Stand To Roll",
  "Standard Walk",
  "Swimming",
  "Thriller Part 3",
  "Treading Water",
  "Walking",
  "Waving",
];

const apath = "./assets/animations/";
manager.onLoad = () => initScene(sceneData);

animations.forEach((name) => {
  loader.load(`${apath}${name}.fbx`, (fbx) => {
    const anim = fbx.animations[0];
    if (anim) {
      anim.name = name;
      sceneData.animations.push(anim);
    }
  });
});

function setupActions(character, animations) {
  const actions = [];
  animations.forEach((anim) => {
    const action = character.userData.mixer.clipAction(anim);
    actions.push(action);
  });
  return actions;
}

// -----------------------------
// Mixamo model support (keeps existing robot code)
// -----------------------------
let mixamoModel = null;

function loadMixamoModel() {
  const gltfLoader = new GLTFLoader();
  gltfLoader.load(
    "./assets/xbot.glb",
    (gltf) => {
      mixamoModel = gltf.scene;
      mixamoModel.position.set(0, -1.3, 0);
      mixamoModel.scale.set(0.9, 0.9, 0.9);
      scene.add(mixamoModel);
      console.log("Mixamo GLB loaded:", gltf);
      statusEl.textContent = "Mixamo XBot loaded";
    },
    undefined,
    (error) => {
      console.warn("Could not load Mixamo GLB:", error);
      statusEl.textContent = "XBot not found — existing robot remains";
    }
  );
}

// -----------------------------
// Mocap Overlay (video pose)
// -----------------------------
let poseLandmarker = null;
let frameSkip = 0;
const PROCESS_EVERY = 4;

const grayCanvas = document.createElement("canvas");
const grayCtx = grayCanvas.getContext("2d", { willReadFrequently: true });

const skeletonGroup = new THREE.Group();
scene.add(skeletonGroup);

const jointMeshes = [];
const lineMeshes = [];

const poseConnections = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [1, 5], [5, 6], [6, 7],
  [1, 9], [9, 10], [10, 11],
  [11, 12], [12, 13], [13, 14],
  [11, 23], [23, 24], [24, 25],
  [23, 12], [9, 5]
];

function buildSkeletonOverlay() {
  for (let i = 0; i < 33; i++) {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 12, 12),
      new THREE.MeshStandardMaterial({
        color: 0x4ed9ff,
        emissive: 0x0a3655,
        metalness: 0.15,
        roughness: 0.35
      })
    );
    mesh.visible = false;
    skeletonGroup.add(mesh);
    jointMeshes.push(mesh);
  }

  poseConnections.forEach(([a, b]) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3()
    ]);
    const material = new THREE.LineBasicMaterial({
      color: 0x8ce9bf,
      transparent: true,
      opacity: 0.92
    });
    const line = new THREE.Line(geometry, material);
    skeletonGroup.add(line);
    lineMeshes.push({ line, a, b });
  });
}

buildSkeletonOverlay();

async function setupPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  statusEl.textContent = "Pose model ready";
}

function getGrayValueAtLandmark(video, landmark) {
  if (!video || !video.videoWidth || !video.videoHeight) return 128;

  const w = video.videoWidth;
  const h = video.videoHeight;

  const x = Math.max(0, Math.min(w - 1, Math.round(landmark.x * w)));
  const y = Math.max(0, Math.min(h - 1, Math.round((1 - landmark.y) * h)));

  grayCanvas.width = w;
  grayCanvas.height = h;

  grayCtx.drawImage(video, 0, 0, w, h);
  const data = grayCtx.getImageData(x, y, 1, 1).data;

  const gray = data[0] * 0.299 + data[1] * 0.587 + data[2] * 0.114;
  return gray;
}

function updateSkeletonFromVideo(landmarks) {
  const t = performance.now() * 0.001;

  landmarks.forEach((lm, i) => {
    if (i >= jointMeshes.length) return;

    const gray = getGrayValueAtLandmark(sourceVideo, lm);
    const z = ((gray / 255) * 3.8 - 1.9) + Math.sin(t + i * 0.25) * 0.15;
    const x = (lm.x - 0.5) * 6.0;
    const y = (1.0 - lm.y) * 5.0;

    jointMeshes[i].position.set(x, y, z);
    jointMeshes[i].visible = true;
  });

  for (let i = landmarks.length; i < jointMeshes.length; i++) {
    jointMeshes[i].visible = false;
  }

  poseConnections.forEach(([a, b], idx) => {
    const p1 = landmarks[a];
    const p2 = landmarks[b];
    if (!p1 || !p2) return;

    const gray1 = getGrayValueAtLandmark(sourceVideo, p1);
    const gray2 = getGrayValueAtLandmark(sourceVideo, p2);

    const z1 = ((gray1 / 255) * 3.8 - 1.9) + Math.sin(t + a * 0.25) * 0.15;
    const z2 = ((gray2 / 255) * 3.8 - 1.9) + Math.sin(t + b * 0.25) * 0.15;

    const v1 = new THREE.Vector3((p1.x - 0.5) * 6.0, (1.0 - p1.y) * 5.0, z1);
    const v2 = new THREE.Vector3((p2.x - 0.5) * 6.0, (1.0 - p2.y) * 5.0, z2);

    const line = lineMeshes[idx].line;
    line.geometry.setFromPoints([v1, v2]);
    line.visible = true;
  });
}

videoInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  sourceVideo.src = url;
  sourceVideo.load();
  videoPanel.style.display = "block";

  sourceVideo.play().catch(() => {
    console.log("Autoplay blocked");
  });
});

function initScene(sceneData) {
  const { character, animations } = sceneData;
  const actions = setupActions(character, animations);

  scene.add(character);

  const radius = 10;
  const geometry = new THREE.CircleGeometry(radius, 32);
  const material = new THREE.MeshStandardMaterial({
    color: 0x001020,
  });
  const plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = Math.PI * -0.5;
  plane.receiveShadow = true;
  plane.position.y = -1.5;
  scene.add(plane);

  const sunLight = new THREE.DirectionalLight(0xffffff, 5);
  sunLight.position.set(2, 4, 3);
  sunLight.castShadow = true;
  scene.add(sunLight);

  const sprites = getLayer({
    hue: 0.58,
    numSprites: 8,
    opacity: 0.2,
    radius: 10,
    size: 24,
    z: -10.5,
  });
  scene.add(sprites);

  let timeElapsed = 0;

  function animate(t = 0) {
    requestAnimationFrame(animate);

    timeElapsed += 0.01;

    character?.userData.update(timeElapsed);

    if (mixamoModel) {
      mixamoModel.rotation.y += 0.003;
    }

    if (poseLandmarker && sourceVideo.readyState >= 2 && !sourceVideo.paused) {
      frameSkip++;
      if (frameSkip % PROCESS_EVERY === 0) {
        const result = poseLandmarker.detectForVideo(sourceVideo, performance.now());
        if (result.landmarks && result.landmarks.length > 0) {
          updateSkeletonFromVideo(result.landmarks[0]);
        }
      }
    }

    renderer.render(scene, camera);
    controls.update();
  }

  let index = 2;
  let previousAction;

  function playRandomAnimationClip() {
    const action = actions[index];
    if (action !== previousAction) {
      previousAction?.fadeOut(1);
      action.reset();
      action.fadeIn(1);
      action.play();
      previousAction = action;
    }
    index = Math.floor(Math.random() * actions.length);
  }

  playRandomAnimationClip();
  animate();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === " ") {
      playRandomAnimationClip();
    }
  });
}

loadMixamoModel();
setupPoseLandmarker().catch((err) => {
  console.error(err);
  statusEl.textContent = "Pose model failed to load";
});
