using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neural_network;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            { 
            //закидываем обучающий набор данных
            //результат              пациент болен - 1
            //                       пациент здоров - 0
            /*
             * неправильная температура - T
             * хороший возраст - A
             * курит - S
             * Правильно питается F
             */
            // наши обучающие данные
                // T  A  S  F                  
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };





            //задаем топологию сети - 4 входных нейрона, 1 выходной, и 2 скрытых(слоя или нейрона - я хз)
            var topology = new Topology(4, 1, 0.1, 2);
            //теперь создаем саму нейросеть 
            var neuralNetwork = new NeuralNetwork(topology);
            //теперь нам надо обучить нейронку 40000 раз (эпох)
            //в эту переменную сохраним среднюю квадратическую ошибку - т.е. наше отклонение от обучения
            var difference = neuralNetwork.Learn(outputs, inputs, 10000);
            //и после обучения нейронки мы сможем проверить ее работоспособность

            var results = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                //будем обращаться к нашей нейронной сети и передавать в нее item 2 и для наглядности сохранять в резулься
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.Predict(row).Output;
                results.Add(res);
            }

            for (int i = 0; i<results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);//округляем до 3  знаков после зап
                var actual = Math.Round(results[i], 2);
                //сравниваем
                Assert.AreEqual(expected, actual);

            }

        }

        [TestMethod()]
        public void RecognizeImagesTest()
        {
            var size = 10;
            //путь к папке с изображениями с паразитами
            var parasitizedPath = @"C:\Users\igork\Desktop\geekbrains\С#\нейросеть\archive\cell_images\Parasitized\";
            var unparasitizedPath = @"C:\Users\igork\Desktop\geekbrains\С#\нейросеть\archive\cell_images\Uninfected\";

            var converter = new PictureConverter();
            //создаем нейронку
            //отсюда получаем колчиество входных нейронов           

            var testParasitizedImageInput = converter.Convert(@"C:\Users\igork\source\repos\Neural network\Neural networkTests\Images\Parasitized.png");
            var testUnparasitizedImageInput = converter.Convert(@"C:\Users\igork\source\repos\Neural network\Neural networkTests\Images\Unparasitized.png");

            var topology = new Topology(testParasitizedImageInput.Length, 1, 0.1, testParasitizedImageInput.Length / 2);
            var neuralNetwork = new NeuralNetwork(topology);

            //получить весь архив изображений в виде масива строк
            double[,] parasitizedInputs = GetData(parasitizedPath, converter, testParasitizedImageInput, size);


            //теперь можно нейронку обучить, это мы обучили по паразитам, поэтому экспектед у нас 1
            neuralNetwork.Learn(new double[] { 1 }, parasitizedInputs, 1);

            //обучили по не паразитам
            double[,] unparasitizedInputs = GetData(unparasitizedPath, converter, testParasitizedImageInput, size);
            neuralNetwork.Learn(new double[] { 0 }, unparasitizedInputs, 1);

            //теперь проверка
            var par = neuralNetwork.Predict(testParasitizedImageInput.Select(t => (double)t).ToArray());
            var unpar = neuralNetwork.Predict(testUnparasitizedImageInput.Select(t => (double)t).ToArray());

            Assert.AreEqual(1, Math.Round(par.Output, 2));
            Assert.AreEqual(0, Math.Round(unpar.Output, 2));

            //нужно сохранять результаты обучения через сериализацию
        }

        private static double[,] GetData(string parasitizedPath, PictureConverter converter, double[] testImageInput, int size)
        {
            //надо получить обучающую выборку, берем имаджес, получаем все изображения в директории
            var images = Directory.GetFiles(parasitizedPath);
            // после этого мы берем сигналы резалт эьто будет количество входных нейронов
            var result = new double[size, testImageInput.Length];

            //и теперь нейронку надо как-нибудь обучить, берем 100 изображений из паки
            for (int i = 0; i < size; i++)
            {
                //взяли изображения, сконвертировали и заполнили массив, который будем использовать для обучения

                //берем 1 изображение из коллекции и начинаем попиксельно сохранять в результат

                var image = converter.Convert(images[i]);
                for (int j = 0; j < image.Length; j++)
                {
                    result[i, j] = image[j];
                }
            }

            return result;
        }

        [TestMethod()]
        public void DatasetTest()
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();
            //сначала читаем файл
            using (var sr = new StreamReader("heart.csv"))
            {
                //1-я строка заголовочная
                var header = sr.ReadLine();

                //прочитали одну строчку и запускаем цикл
                while(!sr.EndOfStream)//пока файл не закончился
                {
                    var row = sr.ReadLine(); //прочитали одну строчку и теперь надо ее распарсить
                    var values = row.Split(',').Select(v => Convert.ToDouble(v.Replace(".", ","))).ToList();//парсим по запятой получаем массив даблов
                    //из этих даблов последнее значение будет являться как раз-таки результатом
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();//взять определенное количество элементов из массива
                    //т.е. берем 1 элемент и закидываем его в 

                    outputs.Add(output);
                    inputs.Add(input);
                }

            }

            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
               for  (var j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }

            //выходных нейронов только 1, обучаемость 0.1, 
            var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count/2);
            //теперь создаем саму нейросеть 
            var neuralNetwork = new NeuralNetwork(topology);
            //теперь нам надо обучить нейронку 40000 раз (эпох)
            //в эту переменную сохраним среднюю квадратическую ошибку - т.е. наше отклонение от обучения
            var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 1000);
            //и после обучения нейронки мы сможем проверить ее работоспособность


            //а теперь проверяем работу нейронки
            var results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                //будем обращаться к нашей нейронной сети и передавать в нее item 2 и для наглядности сохранять в резулься
                
                var res = neuralNetwork.Predict(inputs[i]).Output;
                results.Add(res);
            }

            for (int i = 0; i<results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);//округляем до 3  знаков после зап
                var actual = Math.Round(results[i], 2);
                //сравниваем
                Assert.AreEqual(expected, actual);

            }

        }

    }
}