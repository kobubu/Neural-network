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
    public class PictureConverterTests
    {
        [TestMethod()]
        public void ConvertTest()
        {
            var converter = new PictureConverter();
            var inputs = converter.Convert(@"C:\Users\igork\source\repos\Neural network\Neural networkTests\Images\Parasitized2.png");
            converter.Save(@"C:\Users\igork\source\repos\Neural network\Neural networkTests\Images\converted.png", inputs);
        }
    }
}