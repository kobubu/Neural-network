using System;
using System.Collections.Generic;

namespace Neural_network
{
    //пишем класс нейрон
    public class Neuron
    {
        //есть свойство вес (на списке) - это коэффициенты для умножения входных сигналов
        public List<double> Weights { get; }
        //есть свойство тип
        public NeuronType NeuronType { get; }
        //еще нужен результат куда кладется итог суммирования нейронов
        public double Output { get; private set; }

        //при создании нейрона важно знать, какое количество связей/сигналов к нему будет поступать
        //inputcount - количество входных нейронов
        //по умолчанию тип нейрона нормальный
        public Neuron(int inputСount, NeuronType type = NeuronType.Normal)
        {
            //сюда нужно прописать проверку входных парамнтров и в принципе нужно прописать проверку значения после выполнения метода
            NeuronType = type;
            //по количеству нейронов мы можем отпределить число весов
            Weights = new List<double>();

            for (int i = 0; i < inputСount; i++)
            {
                //по умолчанию заполним еденицей потом перепишем
                Weights.Add(1);
            }

            //метод который выполняет вычисление всех входящих параметро наховем его feedforward  - это линейное распространение слева направо, рекурентные сети, это те, которые ходят внутри себя много раз
        }

        //принимает на вход список входных сигналов, приходящих на нейрон
        public double FeedForward(List<double> inputs)
        {
            //нужно сверится. что количество нейронов и входных весов совпадеет 

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i]*Weights[i];
            }

            Output = Sigmoid(sum);
            return Output;

        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Exp(-x));
            return result;
        }

        //на всякий случай переопределяем метод toString для того, чтобы можно было отлаживать

        public override string ToString()
        {
            return Output.ToString();
        }

        public void SetWeights(params double[] weights)
        {
            //удилить после добавления возможности обучения сети
            for (int i = 0; i<weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        
        }
    }
}
