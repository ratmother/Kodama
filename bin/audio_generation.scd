(

SynthDef.new(\osc_buf_env, {
	arg out = 0, buf, size;
	var env;
	env = EnvGen.kr(Env.perc(size/2, size/2), doneAction:2);
	Out.ar(out, PlayBuf.ar(1, buf, BufRateScale.kr(buf)))

}).add;

~net = NetAddr.new("127.0.0.1", 1050);

~dir = "/home/red/Openframeworks/of_v0.10.1_linux64gcc6_release/apps/Kodama_Alpha_1/Kodama/bin/data/slices/";
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
)
// Read in chunks of 6: ,[, file, morality, dim, size, ],
~osc_resp = {
	inf.do{
		~message.postln;
		~message.size.do{arg i;
			//input.postln;
			if(~message[i] == ~message[1], { // ~message[1] is always [.
				if(~message[i].isNil == false, {
					var attribs = [~message[i+1],~message[i+2],~message[i+3],~message[i+4]];
					attribs.postln;
					~incoming = ~incoming.add([attribs, true]);
				});
			});
			0.01.wait;
			}
	};
}.fork;

// check size by buffer length instead!?

// NO ! Get command prompts from Python.

~clock = {
	inf.do{
		~incoming.size.do{arg i;
			if (~incoming[i][1] == true, {
				// process it
				// load the buffer etc.
				~incoming[i][0].postln;
				var file_name = ~incoming[i][0][0];
				var immortality = ~incoming[i][0][1];
				var dim = ~incoming[i][0][2];
				var size = ~incoming[i][0][3];
				size = size / 44100; // incoming size attrib is in length of buffer, to get seconds we must do this.
				Buffer.read(s, ~dir +/+ file_name, action:{arg buff;
					// buff now refers to the buffer
					buff.size.postln;
					Synth.new(\osc_buf_env,[\buf, buff, size]);
					{// free the buffer after 0.5 secs
						0.5.wait;
						buff.free;
					}.fork;
				}
				);
				~incoming[i][1] = false;
			});
		};
		0.01.wait;
     	};
}.fork;
)

