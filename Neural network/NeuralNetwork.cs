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
        public Neuron FeedForward(List<double> inputSignals)
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

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
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
                var neuron = new Neuron(lastLayer.Count, NeuronType.Output);
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
                var neuron = new Neuron(lastLayer.Count);
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
