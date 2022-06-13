using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


//при нормализации данных мы используем специальные формы, чтобы входные значение все привести к от 0 до 1
//масштабирование на отрезок [0; 1]
//нормализация - выстраивание значений относительно нуля, находится среднее по всем возможным входным значений
//допустим, среднее значение - это 10 тысяч, а минимум - 5 тысяч, 
//для нормализации нужны формулы - находим среднее значение признака по всем ходам
/* 
 * нормализация:
 * https://disk.yandex.ru/i/tTnNWnTEmnBCaw
 * https://youtu.be/3esYbQ9PHrM?t=1140
 * 
 * формула стандартного отклонения признака
 * https://youtu.be/3esYbQ9PHrM?t=1251
 * 
 * формула вычисления значения нового значения нейрона
 * https://youtu.be/3esYbQ9PHrM?t=1339
 * 
 * сперва ищем среднее арифметическое значение по всем, приходящим на 1 нейрон
 * потом вычисляем стандартное отклонение для типа
 * 
 * для масштабирования попроще 
 * для мин и макс есть в линке метод, можно его использовать 
 * https://youtu.be/3esYbQ9PHrM?t=1885
 * 
Обучение нейронной сети выполняется на открытых датасетах с сайта kaggle.com:

https://www.kaggle.com/ronitf/heart-d... - Датасет болезней сердца — эта база данных содержит 76 атрибутов, таких как возраст, пол, тип боли в груди, артериальное давление в покое и другие

https://www.kaggle.com/iarunava/cell-... - Набор данных о клетках малярии — сотовые изображения для выявления малярии
 */


namespace Neural_network
{
    //нейронняа сеть представляет собой коллекцию слоев
    public class NeuralNetwork
    {

        static public void Main()
        {
        }
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            //объявили слои, теперь нужно их реализовывать - т.е. заполнять нейронами
            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }

        //метод, который будет принимать какие-то параметры и осуществлять прогон по нейросети, но не для отдельного нейрона, а всей большой нейросети
        //на вход получает какое-то количество входных сигналов и отправляет на входной уровень
        //при этом количество входных сигналов должно соответствовать колиеству нейронов на сети
        public Neuron FeedForward(params double[] inputSignals)
        {
            //c помощью этого отправляем данные на наши входные нейроны
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();
          
            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }

            else
            {
                //упорядочиваем нейроны с помощью Linq такие, что по аутпуту нисходящему и берем первый их ниъ
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        //метод, принимающий коллекцию для обучения
        public double Learn(double[] expected, double[,] inputs, int epoch) //сразу передаем набор ожидаемых результатов и двумерный массив всех входных сигналов, epoch - количество прогонов всего обучающего датасета
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);
                //берем 1 набор данных и отправляем на обучение
                    //метод Backpropagation возвращает нам нашу ошибку, соответственно мы будем эту ошибку подсчитывать
                    error += Backpropagation(output, input);
                    //прошли необходимое количство эпох и в result возвращаем среднюю ошибку
                }   
            }
            var result = error / epoch;
            return result;
        }

        //делаем метод для того, чтобы взять 1 ряд из двумерного массива

        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row, i];
            return array;
            
        }

        //наши входные сигналы - все входные сигналы сразу со всего датасета
        private double[,] Scalling(double[,] inputs) 
        {
            //делаем результирующий массив, куда будем складывать наши значения после масштабироваться
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                //cейчас нужно вычислить максимальное, минимальное значение для одной колонки (всех сигналов, которые приходят на 1 нейрон)
                var min = inputs[0, column]; //перавый индекс отвечает за номер, второй за колонку
                var max = inputs[0, column];//просто взяли значение с первого

                //внешний цикл идет по столбцам в датасете, а внутренний идет построчно 
                //почему единица, то потому что нулевой мы уже взяли
                
                for (int row = 1; row < inputs.GetLength(0); row++)
                {

                    var item = inputs[row, column];

                    if(item < min)
                    {
                        min = item;
                    }

                    if(item > max)
                    {
                        max = item; 

                    }
                
                }

                var divider = max - min;
                for (int row = 1; row < inputs.GetLength(0); row++)
                {

                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }
            return result;
        }

        //реализуем алгоритм нормализации
        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var sum = 0.0; //нужно вычислить среднее значение сигнала нейрона

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }

                var average = sum / inputs.GetLength(0);

                //нужна переменная для подсчета ошибки
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                //вот здесь стандартное квадратичное отклонение
                var standardError = Math.Sqrt(error/inputs.GetLength(0));

                //последний шаг - вычислить значение выходного элемента
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    //и теперь устанавливаем новые значения нейрона
                    result[row, column] = (inputs[row, column] - average) / standardError;

                }

            }
            return result;
        }

        //добавляем метод обратного распространения ошибки
        //передаем ожидаемый результат и входные сигналы
        private double Backpropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            //вычислили результат
            var difference = actual - expected;
            //запустили балансировку
            foreach(var neuron in Layers.Last().Neurons)
            {
                //обучаем нейрон
                neuron.Learn(difference, Topology.LearningRate);
            }    

            //после этого движемся по слоям
            for(int j = Layers.Count -2; j >= 0; j--)
            {
                var layer = Layers[j];
                //уже обученный слой справа 
                var previousLayer = Layers[ j + 1];
                //начинаем обучать послойно

                for (int i = 0; i < layer.NeuronCount; i++)
                {

                    var neuron = layer.Neurons[i];
                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        //формула вычисления ошибки, подходящая для всех нейронов, кроме крайнего справа слоя
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;//весов несколько, поэтому нужен вес, относящийся к текущему нейрону [i]
                        //и теперь выполняем обучение нейрона на значение этой ошибки
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            //осталось вернуть разницу, а возвращают обычно квадратическую ошибку, поэтому мы также вернем
            var result = difference * difference;
            return result;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                //получаем все сигналы
                var previousLayerSignals = Layers[i - 1].GetSignals();
                //перебираем все нейроны этого слояя и отправляем туда все сигналы с предыдущего слоя
                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);

                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double> { inputSignals[i] };

                //нужен еще отдельный нейрон, на который подаем
                var neuron = Layers[0].Neurons[i];
                neuron.FeedForward(signal);
            }
        }



        //делаем циклы
        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            //получаем крайний слой для того, чтобы определить сколько нейронов там было
            var lastLayer = Layers.Last(); 
            for (int i = 0; i < Topology.OutputCount; i++)
            { //здесь нужно знать, какое количество нейронов на предыдущем слое 
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            //добавляем в качестве первого словя в нашу коллекцию слоев
            Layers.Add(outputLayer);
        }

        //ггенерация скрытых слоев
        private void CreateHiddenLayer()
        {
            //hidden layers.count - это сколько нейронов в каждом из слоев хранится внутри этого параметра
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                //получаем крайний слой для того, чтобы определить сколько нейронов там было
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
            { //здесь нужно знать, какое количество нейронов на предыдущем слое 
                var neuron = new Neuron(lastLayer.NeuronCount);
                hiddenNeurons.Add(neuron);
            }
            var hiddenLayer = new Layer(hiddenNeurons, NeuronType.Output);
            //добавляем в качестве первого словя в нашу коллекцию слоев
            Layers.Add(hiddenLayer);
            }
        }

        //количество нейронов входных сети хранится в топологии

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            { //у input
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            //добавляем в качестве первого словя в нашу коллекцию слоев
            Layers.Add(inputLayer); 
        }
    }
}
