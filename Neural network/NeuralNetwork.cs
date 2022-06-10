using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network
{
    //нейронняа сеть представляет собой коллекцию слоев
    public class NeuralNetwork
    {

        static public void Main(String[] args)
        {

         
        }
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            //объявили слои, теперь нужно их реализовывать - т.е. заполнять нейронами
            
            CreateInputLayers();
            CreateHiddenLayers();
            CreateOutputLayers();

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
        public double Learn(List<Tuple<double, double[]>> dataset, int epoch) //dataset - то, на чем мы обучаем, epoch - количество прогонов всего обучающего датасета
        {
            var error = 0.0;
            for(int i = 0; i < epoch; i++)
            {
                //берем 1 набор данных и отправляем на обучение
                foreach(var data in dataset)
                {
                    //метод Backpropagation возвращает нам нашу ошибку, соответственно мы будем эту ошибку подсчитывать
                    error += Backpropagation(data.Item1, data.Item2);
                }
                //прошли необходимое количство эпох и в result возвращаем среднюю ошибку
            }
            var result = error/epoch;
            return result;
        }
        //добавляем метод обратного распространения ошибки
        //передаем ожидаемый результат и входные сигналы
        private double Backpropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            //вычислили результат
            var difference = actual-expected;
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
                var previousLayer = Layers[j+1];
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
            var result = difference*difference;
            return result;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count;)
            {
                var layer = Layers[i];
                //получаем все сигналы
                var previousLayerSignals = Layers[i-1].GetSignals();
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
            };
        }



        //делаем циклы
        private void CreateOutputLayers()
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
        private void CreateHiddenLayers()
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

        private void CreateInputLayers()
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
