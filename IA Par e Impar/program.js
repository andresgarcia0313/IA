async function loadModel() {//Cargar el modelo de TensorFlow.js
    await tf.ready();//Esperar a que cargue TensorFlow.js
    console.log('TensorFlow.js is ready');//Imprimir que TensorFlow.js está listo
}
loadModel();//Cargar el modelo de TensorFlow.js
// Crear la red neuronal
//Secuencial es una red neuronal que tiene una sola capa de entrada y una sola capa de salida
const model = tf.sequential();
//Dense es una capa de neuronas completamente conectadas
model.add(tf.layers.dense({ units: 2, inputShape: [1] }));
//Sigmoid es una función de activación que toma cualquier número real y lo mapea a un número entre 0 y 1
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
//Compilar la red neuronal
model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam' });

// Entrenar la red neuronal
//tensor2d es una matriz de dos dimensiones para representar los datos de entrada
const xs = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 1]);
//tensor2d es una matriz de dos dimensiones para representar los datos de salida
const ys = tf.tensor2d([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [10, 1]);

//Entrenar la red neuronal
model //fit es el método para entrenar la red neuronal
    .fit(
        xs,//Datos de entrada
        ys,//Datos de salida
        { epochs: 100 }//Número de iteraciones
    )
    .then(//Cuando termine el entrenamiento
        () => {
            // Hacer inferencia con la red neuronal
            //input es el dato de entrada
            const input = tf.tensor2d([10], [1, 1]);
            //output es el dato de salida
            const output = model.predict(input);
            //dataSync[0] obtiene el primer valor del tensor con la salida, indicando si es impar.
            const prediction = output.dataSync()[0];
            //Imprime el resultado
            console.log(prediction);
        }
    );