// This script is for use in Kodama perfmances. It recieves OSC messages containing file paths and other information and plays the audio at these file paths. See the provided Python scripts for more information.

SynthDef.new(\osc_buf_env, {
	arg out = 0, buf, siz = 0.5, mort = 0, dim = 1; // The mort and dim variables correspond to the mortality and dim variables contained in each sample.
	var env, sig;
	env = EnvGen.kr(Env.perc(siz/2 + 1/dim, siz/2), doneAction:2); // Using + 1/dim here to make larger segments of audio have a larger onset.
	sig =  PlayBuf.ar(1, buf, BufRateScale.kr(buf));
	sig = FreeVerb.ar(sig, 0.3 + mort * 100, 1 / dim); // mort * 100 is here because mortality values are typically very small (can be around 0.005) so to make it have an effect on this reverb parameter I needed to do this.
Out.ar([0,1], sig);
}).add;


~net = NetAddr.new("127.0.0.1", 1050);

~dir = "/home/red/Openframeworks/of_v0.10.1_linux64gcc6_release/apps/Kodama_Alpha_1/git2/Kodama/bin/data/slices/";
~incoming = Array.new;


thisProcess.openUDPPort(1050);

OSCdef.new(
	\toggle,
	{
		arg msg, time, addr, port;
		~message = msg;
	},
	'/kodama/'
);


~osc_resp = {
	inf.do{
		~message.size.do{arg i;
			//input.postln;
			if(~message[i] == ~message[1], { // ~message[1] is always [
				if(~message[i].isNil == false, {
					var attribs = [~message[i+1],~message[i+2],~message[i+3],~message[i+4]]; // corresponds to [segment file name, mortality, dim, segment size]
					~incoming = ~incoming.add([attribs, true]);
				});
			});
			0.01.wait;
			}
	};
}.fork;


~clock = {
	inf.do{
		~incoming.size.do{arg i;
			var size;
			if (~incoming[i][1] == true, {
				size = ~incoming[i][0][3]/44100;
				Buffer.read(s, ~dir +/+ ~incoming[i][0][0], action:{arg buff;
					Synth.new(\osc_buf_env,[\buf, buff, \siz, size,\mort,~incoming[i][0][1],\dim,~incoming[i][0][2]]);
					{// Free the buffer after the audio is complete.
						size.wait;
						buff.free;
					}.fork;
				}
				);
				~incoming[i][1] = false;
			});
		};
		1.wait;
     	};
}.fork;


