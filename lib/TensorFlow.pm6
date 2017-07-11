=begin pod

=begin NAME

TensorFlow - Bind TensorFlow to Perl 6

=end NAME

=begin SYNOPSIS

    # Passive TensorFlow
    #
    my $v0 = TensorFlow::Tensor.new( 1, 2 );
    $v0 += 2;
    is $v0.[1], 3;

    # Active TensorFlow
    #
    my $v0 = TensorFlow::Tensor.new;
    my $v1 = TensorFlow::Tensor.new( 0, 0 );
    $v1 += $v0;
    TensorFlow.run( $v0, [ TensorFlow::Tensor.new( 1, 2 ) ] );
    is $v1.[0], 1;
    is $v1.[1], 2;

=end SYNOPSIS

=begin DESCRIPTION

TL;DR version - TensorFlow for Perl 6.

=end DESCRIPTION

=begin METHODS

=end METHODS

=end pod

use Inline::Python;

my role Debugging {
}

my role Testing {

}

my role Validating {

}

#use string:from<Python>;
#say string::capwords('foo bar');

#use numpy:from<Python>;
#use tensorflow:from<Python>;

# Notes for the internals:
#
# For Pete's sake don't use __ in method names, and don't use trailing '_'.
# It's reserved for ... things that confuse python.
#
class TensorFlow {
	also does Debugging;
	also does Testing;
	also does Validating;

	method rebuild {
		my $ip = Inline::Python.new;
		$ip.run( q:to[_END_] );
import numpy as np
import tensorflow as tf
_END_

		$ip.run( q:to[_END_] );
class Shield:
	def get(self):
		return self.value
	def set(self,value):
		self.value = value
class TensorFlow:
	def _Line1(self):
		W = tf.Variable([.3], dtype=tf.float32)
		return W
	def _Line2(self):
		b = tf.Variable([-.3], dtype=tf.float32)
		return b
	def _Line3(self):
		#x = tf.placeholder(tf.float32)
		x = Shield();
		x.set( tf.placeholder(tf.float32) )
		return x
	def _Line4(self, W,x,b):
		#linear_model = W * x + b
		linear_model = W * x.get() + b
		return linear_model
	def _Line5(self):
		y = tf.placeholder(tf.float32)
		return y
	def _Line6(self, linear_model,y):
		loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
		return loss
	def _Line7(self):
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		return optimizer
	def _Line8(self, optimizer,loss):
		train = optimizer.minimize(loss)
		return train
	def _Line9(self):
		x_train = [1,2,3,4]
		return x_train
	def _Line10(self):
		y_train = [0,-1,-2,-3]
		return y_train
	def _Line11(self):
		init = tf.global_variables_initializer()
		return init
	def _Line12(self):
		sess = tf.Session()
		self.Session = sess
		return sess
	def _Line13(self, sess,init):
		#sess.run(init) # reset values to wrong
		self.Session.run(init) # reset values to wrong
#	def _thingie(self, x,x_train, y,y_train):
#		return {x:x_train, y:y_train}
	def _Line14(self, sess,train,x,x_train,y,y_train):
		for i in range(1000):
			#sess.run(train, {x:x_train, y:y_train})
			#self.Session.run(train, {x:x_train, y:y_train})
			self.Session.run(train, {x.get():x_train, y:y_train})
#	def _Line14A(self, sess,train,thingie):
#		print("name ********: %s"%(thingie))
#		for i in range(1000):
#			#sess.run(train, {x:x_train, y:y_train})
#			self.Session.run(train, thingie)
	def _Line15(self, sess,W,b,loss,x,x_train,y,y_train):
		#curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
		#curr_W, curr_b, curr_loss = self.Session.run([W, b, loss], {x:x_train, y:y_train})
		curr_W, curr_b, curr_loss = self.Session.run([W, b, loss], {x.get():x_train, y:y_train})
		return curr_W, curr_b, curr_loss
#	def _Line15A(self, sess,W,b,loss,thingie):
#		#curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
#		curr_W, curr_b, curr_loss = self.Session.run([W, b, loss], thingie)
#		return curr_W, curr_b, curr_loss
	def _Line16(self, curr_W,curr_b,curr_loss):
		print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
_END_

#		my $tf = $ip.call('__main__', 'TensorFlow');

		$ip.run( q:to[_END_] );
my_tf = TensorFlow()
_END_


#		my $W = $tf._Line1();
#		my $b = $tf._Line2();

		# Model parameters
		#
		$ip.run( q:to[_END_] );
W = my_tf._Line1()
b = my_tf._Line2()
_END_

#		my $x = $tf._Line3();
#		my $linear_model = $tf._Line4($W,$x,$b);
#		my $y = $tf._Line5();

		# Model input and output
		#
		$ip.run( q:to[_END_] );
x = my_tf._Line3()
linear_model = my_tf._Line4(W,x,b)
y = my_tf._Line5()
_END_

#		my $loss = $tf._Line6($linear_model,$y);

		# loss
		#
		$ip.run( q:to[_END_] );
loss = my_tf._Line6(linear_model,y)
_END_

#		my $optimizer = $tf._Line7();
#		my $train = $tf._Line8($optimizer,$loss);

		# optimizer
		#
		$ip.run( q:to[_END_] );
optimizer = my_tf._Line7()
train = my_tf._Line8(optimizer,loss)
_END_

#		my $x_train = $tf._Line9();
#		my $y_train = $tf._Line10();

		# training data
		#
		$ip.run( q:to[_END_] );
x_train = my_tf._Line9()
y_train = my_tf._Line10()
_END_

#		my $init = $tf._Line11();
#		my $sess = $tf._Line12();
#		$tf._Line13($sess,$init);
#my $thingie = $tf._thingie($x,$x_train,$y,$y_train);
#		#$tf._Line14($sess,$train,$x,$x_train,$y,$y_train); # XXX ding
#		$tf._Line14A($sess,$train,$thingie);

		# training loop
		#
		$ip.run( q:to[_END_] );
init = my_tf._Line11()
sess = my_tf._Line12()
my_tf._Line13(sess,init)
#thingie = my_tf._thingie(x,x_train,y,y_train)
my_tf._Line14(sess,train,x,x_train,y,y_train)
#my_tf._Line14A(sess,train,thingie)
_END_

##		my ( $curr_W, $curr_b, $curr_loss ) = $tf._Line15($sess,$W,$b,$loss,$x,$x_train,$y,$y_train);
#		my ( $curr_W, $curr_b, $curr_loss ) = $tf._Line15A($sess,$W,$b,$loss,$thingie);
#		$tf._Line16($curr_W,$curr_b,$curr_loss);

		# evaluate training accuracy
		#
		$ip.run( q:to[_END_] );
curr_W, curr_b, curr_loss = my_tf._Line15(sess,W,b,loss,x,x_train,y,y_train)
#curr_W, curr_b, curr_loss = my_tf._Line15A(sess,W,b,loss,thingie)
my_tf._Line16(curr_W,curr_b,curr_loss)
_END_
	}

	method test( ) {
		my $ip = Inline::Python.new;

		$ip.run( q:to[_END_] );
import numpy as np
import tensorflow as tf
_END_

		$ip.run( q:to[_END_] );
class TensorFlow:
	def float32(self):
		return tf.float32
	def Variable(self,value,type):
		return tf.Variable([value], dtype=type)
	def placeholder(self,type):
		return tf.placeholder(type)
	def square(self,value):
		return tf.square(value)
	def reduce_sum(self,value):
		return tf.reduce_sum(value)
	def global_variables_initializer(self):
		return tf.global_variables_initializer()
	def Session(self):
		return tf.Session()

	def _mul_(self,a,b):
		return a.__mul__(b)
	def _add_(self,a,b):
		return a.__add__(b)
	def _sub_(self,a,b):
		return a.__sub__(b)

	def train_GradientDescentOptimizer(self,value):
		return tf.train.GradientDescentOptimizer(value)

	def _run(self,sess,args,dict):
		return sess.run(args,dict)

	def _loop(self,sess,train,x,x_train,y,y_train):
		for i in range(1000):
			#sess.run(train, {x:x_train, y:y_train})
			self._run(sess,train,{x:x_train, y:y_train})

	def _print(self,curr_W,curr_b,curr_loss):
		print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
		
	def run(self,x,y,x_train,y_train,sess,W,b,loss,train):
		# Model parameters
		# W
		# b

		# Model input and output
		# x
		# y
		# linear_model = W * x + b

		# sum of squares
		# loss = self.reduce_sum(self.square(linear_model - y))
		# loss = self.reduce_sum(self.square(linear_model.__sub__(y)))
		# loss = self.reduce_sum(self.square(linear_model_y))
		# loss = self.reduce_sum(square_linear_model_y)

		# training set
		# x_train
		# y_train

		# optimizer
		# optimizer = tf.train.GradientDescentOptimizer(0.01)
		# optimizer = self.train_GradientDescentOptimizer(0.01)
		# train = optimizer.minimize(loss)

		# training loop
		# sess.run(init) # reset values to wrong
		#for i in range(1000):
		#	#sess.run(train, {x:x_train, y:y_train})
		#	self._run(sess,train,{x:x_train, y:y_train})
		#self._loop(1000,sess,train,x,x_train,y,y_train)

		# evaluate training accuracy
		#curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
		curr_W, curr_b, curr_loss = self._run(sess,[W, b, loss], {x:x_train, y:y_train})
		#print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
		self._print(curr_W, curr_b, curr_loss)
_END_

		my $foo = $ip.call('__main__', 'TensorFlow');

		my $float32 = $foo.float32();

		my $x_train = [ 1, 2, 3, 4 ];
		my $y_train = [ 0, -1, -2, -3 ];

		my $W = $foo.Variable(.3, $float32); # XXX [] here.
		my $b = $foo.Variable(-.3, $float32);

		my $x = $foo.placeholder($float32);
		my $y = $foo.placeholder($float32);

		my $linear_model = $foo._add_($foo._mul_($W,$x),$b);
		my $linear_model_y = $foo._sub_($linear_model,$y);
		my $sess = $foo.Session();
		my $init = $foo.global_variables_initializer();
		my $loss = $foo.reduce_sum($foo.square($linear_model_y));
		my $optimizer = $foo.train_GradientDescentOptimizer(0.01);
		my $train = $optimizer.minimize($loss);

		$sess.run($init);

		$foo._loop($sess,$train,$x,$x_train,$y,$y_train);

		$foo.run(
			x => $x,
			y => $y,

			x_train => $x_train,
			y_train => $y_train,

			sess => $sess,

			W => $W,

			b => $b,

			loss => $loss,
			train => $train
		);
	}
}
