using System;
using System.Collections.Generic;

namespace Neural_network
{
    //пишем класс нейрон
    public class Neuron
    {
        //есть свойство вес (на списке) - это коэффициенты для умножения входных сигналов
        public List<double> Weights { get; }
        //свойство для сохранения входных сигналов
        public List<double> Inputs { get; }

        //есть свойство тип
        public NeuronType NeuronType { get; }
        //еще нужен результат куда кладется итог суммирования нейронов
        public double Output { get; private set; }
        //еще нужна дельта, участвующая в вычислениях при обучении
        public double Delta { get; private set; }


        //при создании нейрона важно знать, какое количество связей/сигналов к нему будет поступать
        //inputcount - количество входных нейронов
        //по умолчанию тип нейрона нормальный
        public Neuron(int inputСount, NeuronType type = NeuronType.Normal)
        {
            //сюда нужно прописать проверку входных парамнтров и в принципе нужно прописать проверку значения после выполнения метода
            NeuronType = type;
            //по количеству нейронов мы можем отпределить число весов
            Weights = new List<double>();
            //инициализировали к-во входных сигналов
            Inputs = new List<double>();

            InitWeightsRandomValue(inputСount);

            //метод который выполняет вычисление всех входящих параметро наховем его feedforward  - это линейное распространение слева направо, рекурентные сети, это те, которые ходят внутри себя много раз
        }

        private void InitWeightsRandomValue(int inputСount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputСount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    //в случае входящего нейрона инпут = 1
                    Weights.Add(1);
                }
                else
                {
                    //во всех других случаях у нас случайный коэффициент
                    Weights.Add(rnd.NextDouble());
                }
                
                //прокидываем какие-то значения сигналов
                Inputs.Add(0);
            }
        }

        //принимает на вход список входных сигналов, приходящих на нейрон
        public double FeedForward(List<double> inputs)
        {
            //нужно сверится. что количество нейронов и входных весов совпадеет 

            //cигналы сохраняем
            for (int i = 0; i < inputs.Count; i++)
            {
                inputs[i] = inputs[i];
            }
            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            //здесь проверка на то, что если это не импутный нейрон, то сигмойдная функция не применяется
            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;

        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));  //1.0 / (1.0 + Math.Exp(-x));
            return result;
        }

        //делаем произвоидную от сигмоиды

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }



        //на всякий случай переопределяем метод toString для того, чтобы можно было отлаживать

        public override string ToString()
        {
            return Output.ToString();
        }

        //пишем метод для изменения коэффициентов - то, что будет реализовываться внутри алгоритма и вычислять новые веса 
        //изменяем наш нейрон - на вход подаем разницы - коэффициент, на который нужно изменить коэффициенты
        //по сути это метод для вычисления новых весов
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return; //если тип нейрона - входной, то мы его не обучаем, тк входные нейроны нужны только для проброса сигналов и
                        //выполнения масштабирования

            }
            //в качестве x подставляем текущеее значение нейрона
            Delta = error * SigmoidDx(Output);
            //зная дельту можем начать вычислять веса для каждого из нейронов
            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];
                var newWeight = weight - input * Delta * learningRate;
                //теперь остается присвоить вес
                Weights[i] = newWeight;
            }   
        }
    }
}
