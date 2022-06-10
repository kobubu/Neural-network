using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network
{

    public class Topology
    {
        //количество входов в нейронную сеть (не в нейрон отдельный, а в первый входной слой)
        public int InputCount { get; }

        //количество выходов
        public int OutputCount { get; }

        public double LearningRate { get; }
        //по середине скрытые слои и количество нейронов в них может отличаться
        //коллекция, в которой на каждом слое хранится количство нейронов на этом слое  
        public List<int> HiddenLayers { get; }

        //params - для задания количества нейронов для каждого слоя  
        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            //прокидываем наш массив
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);



        }


    }
}
