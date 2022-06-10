using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neural_network;
using System;
using System.Collections.Generic;
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
            var dataset = new List<Tuple<double, double[]>>
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

            // наши обучающие данные                      T  A  S  F                  
            new Tuple<double, double[]>(0, new double[] { 0, 0, 0, 0 }),
            new Tuple<double, double[]>(0, new double[] { 0, 0, 0, 1 }),
            new Tuple<double, double[]>(1, new double[] { 0, 0, 1, 0 }),
            new Tuple<double, double[]>(0, new double[] { 0, 0, 1, 1 }),
            new Tuple<double, double[]>(0, new double[] { 0, 1, 0, 0 }),
            new Tuple<double, double[]>(0, new double[] { 0, 1, 0, 1 }),
            new Tuple<double, double[]>(1, new double[] { 0, 1, 1, 0 }),
            new Tuple<double, double[]>(0, new double[] { 0, 1, 1, 1 }),
            new Tuple<double, double[]>(1, new double[] { 1, 0, 0, 0 }),
            new Tuple<double, double[]>(1, new double[] { 1, 0, 0, 1 }),
            new Tuple<double, double[]>(0, new double[] { 1, 0, 1, 0 }),
            new Tuple<double, double[]>(1, new double[] { 1, 0, 1, 1 }),
            new Tuple<double, double[]>(1, new double[] { 1, 1, 0, 0 }),
            new Tuple<double, double[]>(1, new double[] { 1, 1, 0, 0 }),
            new Tuple<double, double[]>(0, new double[] { 1, 1, 0, 1 }),
            new Tuple<double, double[]>(1, new double[] { 1, 1, 1, 0 }),
            new Tuple<double, double[]>(1, new double[] { 1, 1, 1, 1 })
            };

        
        


        //задаем топологию сети - 4 входных нейрона, 1 выходной, и 2 скрытых(слоя или нейрона - я хз)
        var topology = new Topology(4, 1, 0.1, 2);
            //теперь создаем саму нейросеть 
            var neuralNetwork = new NeuralNetwork(topology);
            //теперь нам надо обучить нейронку 40000 раз (эпох)
            //в эту переменную сохраним среднюю квадратическую ошибку - т.е. наше отклонение от обучения
            var difference = neuralNetwork.Learn(dataset, 40000);
            //и после обучения нейронки мы сможем проверить ее работоспособность

            var results = new List<double>();
            foreach(var data in dataset)
            {
                //будем обращаться к нашей нейронной сети и передавать в нее item 2 и для наглядности сохранять в резулься
                var res = neuralNetwork.FeedForward(data.Item2).Output;
                results.Add(res);
            }

            for(int i = 0; i<results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 3);//округляем до 3  знаков после зап
                var actual = Math.Round(results[i], 3);
                //сравниваем
                Assert.AreEqual(expected, actual);

            }

        }
    }
}