// Importación de las bibliotecas necesarias
import '@tensorflow/tfjs-backend-cpu'; // Importa el backend de TensorFlow.js para CPU
import '@tensorflow/tfjs-backend-webgl'; // Importa el backend de TensorFlow.js para WebGL

import * as use from '@tensorflow-models/universal-sentence-encoder'; // Importa el modelo Universal Sentence Encoder
import * as tf from '@tensorflow/tfjs-core'; // Importa TensorFlow.js Core
import { interpolateReds } from 'd3-scale-chromatic'; // Importa una función de la biblioteca d3-scale-chromatic

// Conjunto de oraciones de ejemplo para comparar
const sentences = [
  'I like drinking mate.', 'Your cellphone is awesome.', 'How old are you?',
  'What’s your age?', 'I’m from Córdoba, capital.', 'Going to the stadium is the best.'
];

let model; // Variable para almacenar el modelo USE

// Función de inicialización
const init = async () => {
  model = await use.load(); // Carga el modelo USE
  document.querySelector('#loading').style.display = 'none'; // Oculta el elemento de carga
  document.querySelector('#container').style.display = 'block'; // Muestra el contenedor de la aplicación
  
  // Renderiza las oraciones en HTML
  renderSentences();
};

// Función para renderizar las oraciones en HTML
const renderSentences = () => {
  sentences.forEach((sentence, i) => {
    const sentenceDom = document.createElement('div');
    sentenceDom.textContent = `${i + 1}) ${sentence}`;
    document.querySelector('#sentences-container').appendChild(sentenceDom);
  });
};

// Función para comparar la oración ingresada por el usuario con las oraciones de ejemplo
const compareSentence = async () => {
  if (!model) {
    alert('El modelo no se ha cargado aún.'); // Alerta si el modelo no se ha cargado
    return;
  }
  const userSentence = document.getElementById('user-sentence').value; // Obtiene la oración ingresada por el usuario
  if (!userSentence) {
    alert('Por favor, ingresa una oración'); // Alerta si no se ha ingresado ninguna oración
    return;
  }

  try {
    const embeddings = await model.embed([userSentence, ...sentences]); // Incorpora la oración del usuario junto con las oraciones de ejemplo
    const userEmbedding = tf.slice(embeddings, [0, 0], [1, -1]); // Obtiene la incrustación de la oración del usuario
    const resultsContainer = document.getElementById('similarity-scores'); // Obtiene el contenedor de los resultados
    resultsContainer.innerHTML = '<h3>Puntuaciones de similitud:</h3>'; // Agrega un encabezado a los resultados

    for (let i = 0; i < sentences.length; i++) {
      const sentenceEmbedding = tf.slice(embeddings, [i + 1, 0], [1, -1]); // Obtiene la incrustación de una oración de ejemplo
      const score = tf.matMul(userEmbedding, sentenceEmbedding, false, true).dataSync(); // Calcula el producto punto para medir la similitud
      const scoreDiv = document.createElement('div');
      scoreDiv.textContent = `Oración ${i + 1}: ${score[0]}`; // Muestra la puntuación de similitud en el DOM
      resultsContainer.appendChild(scoreDiv); // Agrega el elemento al contenedor de resultados
    }
  } catch (error) {
    console.error('Error comparando oraciones:', error); // Maneja los errores
  }
};

init(); // Inicializa la aplicación cuando se carga la página

// Agrega un event listener al botón de comparación para activar la función de comparación de oraciones
document.getElementById('compare-btn').addEventListener('click', compareSentence);
