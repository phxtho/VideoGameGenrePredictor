import 'babel-polyfill';
import {VG_GenreNet} from './vg_genre_net';

const game_name = document.getElementById('game_name');
game_name.onload = async () => {
  const resultElement = document.getElementById('result');

  resultElement.innerText = 'Loading VG_GenreNet...';

  const model = new VG_GenreNet();
  console.time('Loading of model');
  await model.load();
  console.timeEnd('Loading of model');

  console.time('First prediction');
  let result = model.predict(game_name.innerText);
  const topK = model.getTopKClasses(result, 3);
  console.timeEnd('First prediction');

  resultElement.innerText = '';
  topK.forEach(x => {
    resultElement.innerText += `${x.value.toFixed(3)}: ${x.label}\n`;
  });

  console.time('Subsequent predictions');
  result = model.predict(pixels);
  model.getTopKClasses(result, 2);
  console.timeEnd('Subsequent predictions');

  model.dispose();
};