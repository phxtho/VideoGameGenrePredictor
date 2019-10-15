const MODEL_URL = './web_model/model.json';
const INPUT_NODE_NAME = 'name';
const OUTPUT_NODE_NAME = 'softmax';
const GENRE_CLASSES = {
  0: 'action',
  1: 'adventure',
  2: 'casual',
  3: 'indie',
  4: 'massively_multiplayer',
  5: 'rpg',
  6: 'racing',
  7: 'simulation',
  8: 'sports',
  9: 'strategy'
};

/**
 * @param input un-preprocessed input Array.
 * @return The softmax logits.
 */
predict = (input, model) => {
  return model.execute(
      {[INPUT_NODE_NAME]: input}, OUTPUT_NODE_NAME);
}

getTopKClasses = (logits, topK) => {
  const predictions = tidy(() => {
    return softmax(logits);
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



btn = document.getElementById('btn_predict');

btn.onclick = async () => {
 
  game_name = document.getElementById('game_name').innerText;
 
  const resultElement = document.getElementById('result');

  resultElement.innerText = 'Loading VG_GenreNet...';

  //const model = new VG_GenreNet();
  
  console.time('Loading of model');
  let model = await tf.loadGraphModel(MODEL_URL);
  console.timeEnd('Loading of model');

  console.time('First prediction');
  let result = predict(game_name.innerText, model);
  const topK = getTopKClasses(result, 3);
  console.timeEnd('First prediction');

  resultElement.innerText = '';
  topK.forEach(x => {
    resultElement.innerText += `${x.value.toFixed(3)}: ${x.label}\n`;
  });

  console.time('Subsequent predictions');
  result = predict(pixels, model);
  getTopKClasses(result, 2);
  console.timeEnd('Subsequent predictions');

  model.dispose();
};