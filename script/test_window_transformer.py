__author__ = 'KimTS'
from fuel.datasets.youtube_audio import YouTubeAudio
from data.window import Window
import numpy

if __name__=='__main__':
    youtube_id = 'XqaJ2Ol5cC4'
    data = YouTubeAudio(youtube_id)
    stream = data.get_example_stream()
    stream = Window(offset=1,
                    source_window=1000,
                    target_window=1000,
                    overlapping=True,
                    data_stream=stream)
    it = stream.get_epoch_iterator()
    for b, i in enumerate(it):
        print len(i)
        print i[0][:5]
        print i[1][:5]

        raw_input()
