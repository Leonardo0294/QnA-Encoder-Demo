import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-core';
import { interpolateReds } from 'd3-scale-chromatic';

const sentences = [
  'I like my phone.', 'Your cellphone looks great.', 'How old are you?',
  'What is your age?', 'An apple a day, keeps the doctors away.',
  'Eating strawberries is healthy.'
];

let model;

const init = async () => {
  model = await use.load();
  document.querySelector('#loading').style.display = 'none';
  document.querySelector('#container').style.display = 'block';
  
  // Render sentences in HTML
  renderSentences();
};

// Function to render sentences in HTML
const renderSentences = () => {
  sentences.forEach((sentence, i) => {
    const sentenceDom = document.createElement('div');
    sentenceDom.textContent = `${i + 1}) ${sentence}`;
    document.querySelector('#sentences-container').appendChild(sentenceDom);
  });
};

const compareSentence = async () => {
  if (!model) {
    alert('Model is not loaded yet.');
    return;
  }
  const userSentence = document.getElementById('user-sentence').value;
  if (!userSentence) {
    alert('Please enter a sentence');
    return;
  }

  try {
    const embeddings = await model.embed([userSentence, ...sentences]);
    const userEmbedding = tf.slice(embeddings, [0, 0], [1, -1]);
    const resultsContainer = document.getElementById('similarity-scores');
    resultsContainer.innerHTML = '<h3>Similarity Scores:</h3>';

    for (let i = 0; i < sentences.length; i++) {
      const sentenceEmbedding = tf.slice(embeddings, [i + 1, 0], [1, -1]);
      const score = tf.matMul(userEmbedding, sentenceEmbedding, false, true).dataSync();
      const scoreDiv = document.createElement('div');
      scoreDiv.textContent = `Sentence ${i + 1}: ${score[0]}`;
      resultsContainer.appendChild(scoreDiv);
    }
  } catch (error) {
    console.error('Error comparing sentence:', error);
  }
};

init();

document.getElementById('compare-btn').addEventListener('click', compareSentence);
