import { createCubePreview } from './cube3d.js';
import { renderMoveList } from './moveList.js';

const statusEl = document.getElementById('status');
const movesEl = document.getElementById('moves');
const resetBtn = document.getElementById('resetBtn');
const solveBtn = document.getElementById('solveBtn');
const cubeEl = document.getElementById('cube');
const previewEl = document.getElementById('cameraPreview');
const colorPickerEl = document.getElementById('colorPicker');
const facePickerEl = document.getElementById('facePicker');

let cubePreview;
let selectedColor = 'red';
let selectedFace = 'U';
let faceSelection = {};

function log(message) {
  statusEl.textContent = message;
}

cubePreview = createCubePreview(cubeEl);

function updateColorButtons() {
  Array.from(colorPickerEl.children).forEach((child) => {
    child.style.border = child.textContent === colorOptions.find((option) => option.name === selectedColor)?.label ? '2px solid #fff' : '2px solid transparent';
  });
}

function updateFaceButtons() {
  Array.from(facePickerEl.children).forEach((child) => {
    child.style.border = child.textContent === faceOptions.find((option) => option.name === selectedFace)?.label ? '2px solid #fff' : '2px solid transparent';
  });
}

const colorOptions = [
  { name: 'red', label: 'Red' },
  { name: 'orange', label: 'Orange' },
  { name: 'yellow', label: 'Yellow' },
  { name: 'green', label: 'Green' },
  { name: 'blue', label: 'Blue' },
  { name: 'white', label: 'White' },
];

const faceOptions = [
  { name: 'U', label: 'U' },
  { name: 'R', label: 'R' },
  { name: 'F', label: 'F' },
  { name: 'D', label: 'D' },
  { name: 'L', label: 'L' },
  { name: 'B', label: 'B' },
];

colorOptions.forEach((option) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = option.label;
  button.style.background = option.name;
  button.style.color = option.name === 'yellow' ? '#111' : '#fff';
  button.style.border = selectedColor === option.name ? '2px solid #fff' : '2px solid transparent';
  button.addEventListener('click', () => {
    selectedColor = option.name;
    updateColorButtons();
    log(`Selected ${option.label.toLowerCase()} color`);
  });
  colorPickerEl.appendChild(button);
});

faceOptions.forEach((option) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = option.label;
  button.style.border = selectedFace === option.name ? '2px solid #fff' : '2px solid transparent';
  button.addEventListener('click', () => {
    selectedFace = option.name;
    updateFaceButtons();
    cubePreview?.setSelectedFace(selectedFace);
    log(`Selected ${option.name} face`);
  });
  facePickerEl.appendChild(button);
});

function startCamera() {
  previewEl.style.display = 'none';
  log('Offline mode ready. Select a color and paint the cube.');
}

function paintFace(faceName) {
  if (!faceName) {
    return;
  }
  const colors = Array(9).fill(selectedColor);
  cubePreview?.updateFromScan(faceName, colors);
  faceSelection[faceName] = colors;
  log(`Painted ${faceName} face with ${selectedColor}`);
}

function paintSelectedFace() {
  paintFace(selectedFace);
}

function resetScan() {
  faceSelection = {};
  cubePreview?.updateFromScan('U', Array(9).fill('unknown'));
  cubePreview?.updateFromScan('R', Array(9).fill('unknown'));
  cubePreview?.updateFromScan('F', Array(9).fill('unknown'));
  cubePreview?.updateFromScan('D', Array(9).fill('unknown'));
  cubePreview?.updateFromScan('L', Array(9).fill('unknown'));
  cubePreview?.updateFromScan('B', Array(9).fill('unknown'));
  log('Scan reset');
}

function solve() {
  renderMoveList(movesEl, [{ notation: 'Offline demo only' }]);
  log('Offline mode: no solver required.');
}

resetBtn.addEventListener('click', resetScan);
solveBtn.addEventListener('click', solve);

cubePreview?.setSelectedFace(selectedFace);
cubePreview?.setOnFaceSelected((faceName) => {
  selectedFace = faceName;
  updateFaceButtons();
  cubePreview?.setSelectedFace(selectedFace);
  log(`Selected ${faceName} face`);
});

const paintButton = document.createElement('button');
paintButton.type = 'button';
paintButton.textContent = 'Paint selected face';
paintButton.addEventListener('click', paintSelectedFace);
facePickerEl.insertAdjacentElement('afterend', paintButton);

startCamera();
