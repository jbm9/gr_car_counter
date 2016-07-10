#!/usr/bin/env python2

# A sorta-simple car counter in GNU Radio, using a USB sound device
# for input.
#
# Copyright (c) 2016 Josh Myer <josh@joshisanerd.com>
# License: cc-by
#

# For theory of operation in detail, see the Explanation.ipynb that
# should have come with this.  If you don't have that, here's a quick
# run-down:
#
# We're looking for white noise.  To find it, we take FFTs of chunks
# of audio, and grab the median power level.  We smooth these medians
# over a small span of time to get the noise out, then treat them as a
# statistical distribution.  Once the power level we observe is some
# number of sigma above the mean, we start counting.  So long as it
# stays above that for a certain holdoff time, we count it as a car.
# If it's a short burst, though, we don't count it.

# Knobs to twiddle

# The USB sound device name: This is annoying and I can't help you
# much.  See 'arecord -L' for possible values to try shoving in here.
# I'd apologize on ALSA's behalf, but that wouldn't help anybody feel
# better.

from gnuradio import audio
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from optparse import OptionParser

import numpy
import time
import sys
from gnuradio import fft
from gnuradio.gr.gateway import decim_block, basic_block, sync_block



class reduce_block(decim_block):
    "A generic reduce-style block, vec_len items to 1."
    def __init__(self, name, vec_len, fn):
        decim_block.__init__(
            self,
            name = name,
            in_sig  = [ numpy.float32 ],
            out_sig = [ numpy.float32 ],
            decim = vec_len
        )

        self.vec_len = vec_len
        self.fn = fn

    def work(self, input_items, output_items):
        i = 0

        for a in input_items:
            if len(a) % self.vec_len != 0:
                raise Exception("Bogus modulus in vector input to %s: %d" % (name, len(a)))

            for j in range(0, len(a), self.vec_len):
                retval = self.fn(a[j:j+self.vec_len])
                output_items[0][i] = retval
                i += 1
        #print " ==> %d " %i
        return i

class np_median(reduce_block):
    "Returns the median of a vec_length number of entries"
    def __init__(self, vec_len):
        reduce_block.__init__(self, "np_median", vec_len, numpy.median)

class np_mean(reduce_block):
    "Returns the mean of a vec_length number of entries"

    def __init__(self, vec_len):
        reduce_block.__init__(self, "np_mean", vec_len, numpy.mean)


class np_std(reduce_block):
    "Returns the standard deviation of a vec_length number of entries"
    def __init__(self, vec_len):
        reduce_block.__init__(self, "np_std", vec_len, numpy.std)


class jbm_z_score(decim_block):
    """Implements a rolling z-score, based on the mean/std from the last N samples

    N: number of historical samples to use as the sample
    mean: initial mean value
    std: initial standard deviation
    initial_value: value to return until we have good data
    
    Until it's gotten N samples in, it returns initial_value.

    Input ports:
    0: the floats to track
    
    Output ports:
    0: the z-score
    1: the mean at this point
    2: the std at this point

    You probably want to just follow this with a skip_head block.
    
    Internally, this tracks the variance using an algorithm from:
       http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/    
    """
    def __init__(self, N, mean=0.0, std = 1.0, initial_value=0.0):
        decim_block.__init__(
            self,
            name = "Rolling Z-Score",
            in_sig  = [ numpy.float32 ],
            out_sig = [ numpy.float32, numpy.float32, numpy.float32 ],
            decim = 1
        )

        self.N = N
        self.mean = mean
        self.variance = std*std

        self.initial_value = initial_value
        self.values = [initial_value] * N
        self.vcursor = 0
        self.primed = False

    def work(self, input_items, output_items):
        mean = self.mean
        var = self.variance
        std = numpy.sqrt(var)

        emitted = 0

        for i,a in enumerate(input_items):
            for j, x in enumerate(a):
                if not self.primed:
                    output_items[0][j] = self.initial_value
                    output_items[1][j] = self.mean
                    output_items[2][j] = numpy.sqrt(self.variance)

                    #print x, "not primed", self.N
                else:                    
                    output_items[0][j] = (x-mean)/std
                    output_items[1][j] = mean
                    output_items[2][j] = std
                    #print x, mean, var

                oldmean = mean
                x_old = self.values[self.vcursor]

                delta = x - x_old

                mean += delta/self.N
                
                var += delta*(x-mean + x_old-oldmean)/(self.N-1)
                std = numpy.sqrt(var)



                self.values[self.vcursor] = x
                self.vcursor = (self.vcursor + 1) % self.N

                if not self.primed and 0 == self.vcursor:
                    self.mean = numpy.mean(self.values)
                    self.var = numpy.var(self.values)
                    mean = self.mean
                    var = self.var
                    std = numpy.sqrt(var)
                    self.primed = True

                emitted += 1

        self.mean = mean
        self.variance = var

        return emitted

class jbm_threshold_timestamp(decim_block):
    """Prints timestamps to a log file, given a threshold and holdoff

    Fs: Sample rate of the input stream
    threshold: the cutoff to gate at: values above this may trigger
    holdoff: the time (in seconds) to hold off before triggering
    output_fd: a file-like object to write() output to

    Input ports:
    0: the input to be thresholded

    Output ports:
    None.

    This looks for values that above the threshold: once it sees one
    above the threshold, it triggers internally, and starts looking
    for values below the threshold.  Once the value goes below the
    threshold, it looks at how long it was high.  If it was longer
    than holdoff seconds, it will emit a line of the form "timestamp
    duration" to the logfile_fd.

    It also prints lots of crap to stdout.  Hope that helps?
          
    """
    def __init__(self, t0, Fs, threshold, holdoff, output_fd):
        decim_block.__init__(
            self,
            name = "Threshold timestamp with holdoff",
            in_sig  = [ numpy.float32 ],
            out_sig = [ ],
            decim = 1
        )

        self.t0 = t0
        self.t = t0

        self.Fs = Fs
        self.threshold = threshold
        self.holdoff = holdoff
        self.output_fd = output_fd

        self.in_hit = False
        self.hit_start = None


    def work(self, input_items, output_items):
        for x in input_items[0]:
            self.t += 1.0/self.Fs

            if self.in_hit and (x < self.threshold):
                dt = self.t - self.hit_start
                if dt >= self.holdoff:
                    self.output_fd.write("%d %0.02f\n" % (self.hit_start, dt))
                    print("%d %0.02f" % (self.hit_start, dt))
                    self.output_fd.flush()
                else:
                    print "Skipping short run, %d: %0.02f" % (self.t, dt)
                self.in_hit = False
                self.hit_start = None

            elif not self.in_hit and x >= self.threshold:
                self.hit_start = self.t
                self.in_hit = True

        return len(input_items[0])


class jbm_print_decim(decim_block):
    "Quick debug printfs: every N samples, prints the three input ports' values to the screen."
    def __init__(self, N):
        decim_block.__init__(
            self,
            name = "Debug printout of state",
            in_sig  = [ numpy.float32, numpy.float32, numpy.float32 ],
            out_sig = [ ],
            decim = N
        )

        self.N = N

        self.seen = 0

    def work(self, input_items, output_items):
        emitted = 0

        for i in xrange(self.N-self.seen, len(input_items[0]), self.N):
	    print "%f %f %f %f" % (time.time(), input_items[0][i], input_items[1][i], input_items[2][i])
            emitted += 1

        self.seen += len(input_items[0])
        self.seen = self.seen % self.N

        return emitted


class top_block(gr.top_block):

    def __init__(self,
                              alsa_dev="default:CARD=Device",
                              averaging_period=1.0,
                              zscore_period=300.0,
                              trigger_threshold=1.0,
                              trigger_holdoff=2.0,
                              debug_print_period=3.0,
                              logfile_path=None,
                              NFFT = 128,
                              audio_rate=48000,
                              audio_decim=6):
        gr.top_block.__init__(self)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = audio_rate/audio_decim

        ##################################################
        # Blocks
        ##################################################

        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
                interpolation=1,
                decimation=audio_decim,
                taps=None,
                fractional_bw=None,
        )
        self.blocks_multiply_xx_0 = blocks.multiply_vff(1)
        self.audio_source_0 = audio.source(audio_rate, alsa_dev, True)

        self.pack_for_fft = blocks.stream_to_vector(gr.sizeof_float*1, NFFT)
        self.fft = fft.fft_vfc(NFFT, True, (fft.window.blackmanharris(NFFT)), 1)
        self.unpack_fft = blocks.vector_to_stream(gr.sizeof_gr_complex*1, NFFT)

        self.mag = blocks.complex_to_mag(1)

        self.median = np_median(NFFT)


        averaging_samples = int(averaging_period * self.samp_rate / NFFT)
        self.moving_average = blocks.moving_average_ff(averaging_samples, 1.0/averaging_samples)

        z_score_samples = int(zscore_period * self.samp_rate / NFFT)
        self.z_score = jbm_z_score(z_score_samples)

        debug_print_samples = int(debug_print_period * self.samp_rate/NFFT)
        self.print_decim = jbm_print_decim(debug_print_samples)

        if logfile_path == None:
            logfile_path = ("carlog_%d.txt" % time.time())
        logfile = file(logfile_path, "a")
        self.threshold_ts = jbm_threshold_timestamp(time.time(), self.samp_rate/NFFT, trigger_threshold, trigger_holdoff, logfile)




        ##################################################
        # Connections
        ##################################################
        self.connect((self.audio_source_0, 0), (self.rational_resampler_xxx_0, 0))    
        self.connect(self.rational_resampler_xxx_0, self.pack_for_fft)
        self.connect(self.pack_for_fft, self.fft, self.unpack_fft, self.mag)
        self.connect(self.mag, self.median)
        self.connect(self.median, self.moving_average, self.z_score)
        self.connect((self.z_score,0), self.threshold_ts)

        if debug_print_period != 0:
            for i in range(3):
                self.connect((self.z_score, i), (self.print_decim, i))
        else:
            for i in range(3):
                dummy_null_sink = blocks.null_sink(gr.sizeof_float)
                self.connect((self.z_score, i), dummy_null_sink)
            




    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate


if __name__ == '__main__':
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")

    carlog_path = "carlog_%d.txt" % int(time.time())

    parser.add_option("-d", "--audio-device", help="Audio device to use (Linux: see 'arecord -L' for options)", default="default:CARD=Device")    

    parser.add_option("-l", "--logfile-path", help="Path to file to output log into", default=carlog_path)
    
    parser.add_option("-a", "--averaging-period", help="Smoothing period for medians (seconds)", default=1.0)
    parser.add_option("-z", "--zscore-period", help="Span of time to use when computing z-scores (seconds)", default=300.0)
    
    parser.add_option("-t", "--trigger-threshold", help="Trigger threshold (standard deviations over mean)", default=0.8)
    parser.add_option("-o", "--trigger-holdoff", help="Trigger holdoff before latching (seconds)", default=1.0)
    
    parser.add_option("-x", "--debug-period", help="Time between debug printfs, 0 for don't (seconds)", default=3.0)
    
    # May the gods help you if you need these options...

    parser.add_option("-N", "--NFFT", help="Number of FFTs to use (probably don't mess with this...)", default=128)
    
    parser.add_option("-A", "--audio-rate", help="Sample rate to request from audio card (Hz)", default=48000)
    parser.add_option("-D", "--audio-decim", help="Audio decimation rate (before FFTs)", default=6)
    
    (options, args) = parser.parse_args()
    
    tb = top_block(alsa_dev=options.audio_device,
                   averaging_period=float(options.averaging_period),
                   zscore_period=float(options.zscore_period),
                   trigger_threshold=float(options.trigger_threshold),
                   trigger_holdoff=float(options.trigger_holdoff),
                   debug_print_period=float(options.debug_period),
                   logfile_path=options.logfile_path,
                   NFFT=int(options.NFFT),
                   audio_rate=int(options.audio_rate),
                   audio_decim=int(options.audio_decim))
    tb.start()
    tb.wait()
