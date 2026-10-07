'use strict';
// Illustration of the existing presets; these controls never call the backend.
function updateExplanation() {
  const quick = document.getElementById('explanation-quality').value === 'quick';
  const folds = quick ? 3 : 5;
  const kernels = document.getElementById('explanation-bank').value === 'extended' ? 8 : 4;
  const values = {
    comparison: `${kernels} noyaux candidats · les mêmes ${folds} plis pour chaque noyau`,
    folds: `en ${folds} plis de validation`,
    'fold-training': 'Sur les autres plis',
    loop: `${folds} plis`,
    'compact-loop': `Répéter pour chacun des ${folds} plis`,
    'fold-number': folds, 'cv-restarts': quick ? 0 : 2, 'restart-number': quick ? 1 : 10,
  };
  document.querySelectorAll('[data-explanation]').forEach(element => {
    element.textContent = values[element.dataset.explanation];
  });
  document.getElementById('explanation-fit-count').textContent = `${kernels} noyaux × ${folds} plis = ${kernels*folds} ajustements de validation, puis 1 ajustement final.`;
}
document.getElementById('explanation-quality').addEventListener('change', updateExplanation);
document.getElementById('explanation-bank').addEventListener('change', updateExplanation);
updateExplanation();

// These fixed examples are generated with scikit-learn; no training is triggered here.
const kernelExamples = [
  {id:'rbf', name:'RBF', summary:'Une interpolation très lisse, avec des transitions régulières.',
    formula:'k(x, x′) = exp(−r² / 2)', parameters:'r = |x − x′| / ℓ · ℓ = 0,85',
    reading:'La longueur ℓ contrôle la vitesse à laquelle la courbe peut varier.'},
  {id:'matern32', name:'Matérn 3/2', summary:'Des variations plus locales et moins lisses que celles du RBF.',
    formula:'k(x, x′) = (1 + √3 r) exp(−√3 r)', parameters:'r = |x − x′| / ℓ · ℓ = 0,85',
    reading:'Il autorise des fonctions moins régulières. La moyenne du GP peut néanmoins paraître lisse.'},
  {id:'matern52', name:'Matérn 5/2', summary:'Une régularité intermédiaire entre Matérn 3/2 et RBF.',
    formula:'k(x, x′) = (1 + √5 r + 5r² / 3)<br>× exp(−√5 r)', parameters:'r = |x − x′| / ℓ · ℓ = 0,85',
    reading:'Il conserve des variations locales tout en imposant plus de régularité que Matérn 3/2.'},
  {id:'rq', name:'RationalQuadratic', summary:'Un mélange de longueurs de variation, pour combiner plusieurs échelles.',
    formula:'k(x, x′) = (1 + r² / (2α))<sup>−α</sup>', parameters:'r = |x − x′| / ℓ · ℓ = 0,85 · α = 0,7',
    reading:'Le paramètre α règle le mélange des échelles. Lorsque α devient très grand, ce noyau tend vers le RBF.'},
  {id:'linear', name:'Linéaire', summary:'Une droite pour représenter la tendance globale.',
    formula:'k(x, x′) = c² + x x′', parameters:'c = 1',
    reading:'Le modèle est limité aux fonctions affines : une droite ne peut pas suivre toutes les variations des observations.'},
  {id:'quadratic', name:'Quadratique', summary:'Une tendance polynomiale de degré au plus deux.',
    formula:'k(x, x′) = (c² + x x′)²', parameters:'c = 1',
    reading:'Il peut produire une parabole, avec une courbure globale plutôt que des variations locales indépendantes.'},
  {id:'linear-matern', name:'Linéaire + Matérn 3/2', summary:'Une tendance globale enrichie de variations locales.',
    formula:'k(x, x′) = c² + x x′<br>+ (1 + √3 r) exp(−√3 r)', parameters:'r = |x − x′| / ℓ · ℓ = 0,85 · c = 1',
    reading:'La somme des noyaux permet de combiner une tendance affine et une composante moins régulière.'},
  {id:'two-rbf', name:'Deux RBF', summary:'Deux échelles de variation dans une même courbe lisse.',
    formula:'k(x, x′) = exp(−r₁² / 2)<br>+ b² exp(−r₂² / 2)', parameters:'r₁ = |x − x′| / ℓ₁ · r₂ = |x − x′| / ℓ₂\nℓ₁ = 0,85 · ℓ₂ = 2,4 · b² = 0,5',
    reading:'Une composante suit les variations rapides et l’autre les variations lentes. Leur somme reste lisse.'},
];
let currentKernel = 0;
const kernelChoices = document.getElementById('kernel-choices');
const choiceButtons = kernelExamples.map((kernel, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = String(index + 1);
  button.setAttribute('aria-label', `Afficher le noyau ${kernel.name}`);
  button.addEventListener('click', () => showKernel(index));
  kernelChoices.appendChild(button);
  return button;
});
function showKernel(index) {
  currentKernel = (index + kernelExamples.length) % kernelExamples.length;
  const kernel = kernelExamples[currentKernel];
  document.getElementById('kernel-name').textContent = kernel.name;
  document.getElementById('kernel-summary').textContent = kernel.summary;
  // Formula markup comes exclusively from the local, fixed definitions above.
  document.getElementById('kernel-formula').innerHTML = kernel.formula;
  document.getElementById('kernel-parameters').textContent = kernel.parameters;
  document.getElementById('kernel-reading').textContent = kernel.reading;
  document.getElementById('kernel-position').textContent = `${currentKernel + 1} / ${kernelExamples.length}`;
  const image = document.getElementById('kernel-example');
  image.src = `/static/kernel-examples/${kernel.id}.svg`;
  image.alt = `Exemple 1D du noyau ${kernel.name} : observations, moyenne prédite et intervalle à 95 %`;
  choiceButtons.forEach((button, n) => button.setAttribute('aria-pressed', String(n === currentKernel)));
}
document.getElementById('kernel-previous').addEventListener('click', () => showKernel(currentKernel - 1));
document.getElementById('kernel-next').addEventListener('click', () => showKernel(currentKernel + 1));
document.querySelector('.kernel-gallery').addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    showKernel(currentKernel + (event.key === 'ArrowLeft' ? -1 : 1));
  }
});
showKernel(0);
