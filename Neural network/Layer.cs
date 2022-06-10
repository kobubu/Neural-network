using System.Collections.Generic;


namespace Neural_network
{

    //слой - это набор нейронов
   public class Layer
    {
        //слой - это набор нейронов
        public List<Neuron> Neurons { get; }
        //свойство, которое позволяет вычислить количство нейронов, в коде встроена проверка на null
        public int NeuronCount => Neurons?.Count ?? 0;//каунт. такой что если neurons = null? то count = null.

        public NeuronType Type;

        //делаем контсруктор, внутри одного слоя будет список нейронов, котороые нужно првоерить на корректность и так же тип - т.к. на одном слое могут быть нейроны только одного типа
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            //todo проверить все входные нейроны на соответствие типу
            Neurons = neurons;

        }

        //метод, который пройдется по всем нейронам слоя, соберет все сигналы со слоя и по необходимости передавать
        public List<double> GetSignals()
        {

            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        //для слоя переопределяем функцию toString, чтобы она возвращала тип, ибо оно нечитаемое

        public override string ToString()
        {
            //возвращаем тип, чтобы было читаемее при отладке
            return Type.ToString();
        }

    }
}
