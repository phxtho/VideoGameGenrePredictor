import * as tf from '@tensorflow/tfjs';

import {GENRE_CLASSES} from './genre_classes';


const MODEL_URL = '../tensorflow/web_model/model.json';
const INPUT_NODE_NAME = 'name';
const OUTPUT_NODE_NAME = 'softmax';

export class VG_GenreNet {
  constructor() {}

  async load() {
    this.model = await tf.loadGraphModel(MODEL_URL);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
  /**
   * @param input un-preprocessed input Array.
   * @return The softmax logits.
   */
  predict(input) {
    return this.model.execute(
        {[INPUT_NODE_NAME]: input}, OUTPUT_NODE_NAME);
  }

  getTopKClasses(logits, topK) {
    const predictions = tf.tidy(() => {
      return tf.softmax(logits);
    });

    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList
                         .sort((a, b) => {
                           return b.value - a.value;
                         })
                         .slice(0, topK);

    return predictionList.map(x => {
      return {label: GENRE_CLASSES[x.index], value: x.value};
    });
  }
}