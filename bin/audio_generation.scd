(

SynthDef.new(\osc_buf_env, {
	arg out = 0, buf, siz = 0.5;
	var env, sig;
	env = EnvGen.kr(Env.perc(siz/2, siz/2), doneAction:2);
	sig =  PlayBuf.ar(1, buf, BufRateScale.kr(buf));
	sig = FreeVerb.ar(sig, MouseX.kr(), MouseY.kr());
Out.ar(out, sig);
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
		//~incoming = Array.new;
	},
	'/kodama/'
);


~osc_resp = {
	inf.do{
		//~message.postln;
		~message.size.do{arg i;
			//input.postln;
			if(~message[i] == ~message[1], { // ~message[1] is always [.
				if(~message[i].isNil == false, {
					var attribs = [~message[i+1],~message[i+2],~message[i+3],~message[i+4]];
					//attribs.postln;
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
				//~incoming[i][0][0].postln;
				size = ~incoming[i][0][3]/44100;
				// Incoming size attrib is in length of buffer, to get seconds we must do this.
				Buffer.read(s, ~dir +/+ ~incoming[i][0][0], action:{arg buff;
					Synth.new(\osc_buf_env,[\buf, buff, \siz, size]);
					{// Free the buffer after the audio is complete.
						size.wait;
						buff.free;
					}.fork;
				}
				);
				~incoming[i][1] = false; // Can we just delete this?
			});
		};
		1.wait; // interval variable osc 60/bpm
     	};
}.fork;
)

