﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network
{
    public class PictureConverter
    {
        //нужно чтобы принимал на вход путь до изображения принимает путь к файлу изображений

        //обучение без учителя работает по принципу классификатора, т.е. нейронка определяет какие-то закономерности и просто
        //относит входные данные к одной из групп, не зная, что конкретно там есть, не давая какого-то результата

        public int Boundary { get; set; } = 128;
        //cвойство для сохранения высоты
        public int Height { get; set; }
        public int Width { get; set; }
        public List<int> Convert (string path)
        {
            
                var result = new List<int>();
                //нужно прочитать изображение с диска

                var image = new Bitmap(path);
            Height = image.Height;
            Width = image.Width;

           
            //и теперь нужно брать по 1 пикселю из изображения 
            for (int y = 0; y < image.Height; y++)
                {
                for(int x = 0; x < image.Width; x++)
                    { 
                    var pixel = image.GetPixel(x, y);
                    var value = Brightness(pixel);
                    result.Add(value);
                    }
                }
                return result;
             

        }
        //и возвращал список значений, чтобы потом подавать на вход нейронке

        private int Brightness(Color pixel)
        {
        
            var result = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;

            //если резалт ниже порогового значения, то возвращаем 0, иначе 1
        return result < Boundary ? 0 : 1;
        }

        //еще 1 метод, чтобы мы могли преобразовать пиксели в картинку и по необходимости поменять пороговое значение

        //string path - куда сохранять изображение, высоту и ширину изображения
        public void Save(string path, List<int> pixels)
        {
            var image = new Bitmap(Width, Height);
            
                for (int y = 0; y < image.Height; y++)
                {
                for (int x = 0; x < image.Width; x++)
                    {
                    var color = pixels[y * Width + x] == 1 ? Color.White : Color.Black; 
                    image.SetPixel(x, y, color);
                    }
                }
            image.Save(path);
        }



        //if (File.Exists(path))
        //{ }
        //else 
        //{
        //    throw new ArgumentException("File not found");
        //}
    }
}
