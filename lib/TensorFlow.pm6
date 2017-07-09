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

class TensorFlow {
	also does Debugging;
	also does Testing;
	also does Validating;

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
		for i in range(1000):
			#sess.run(train, {x:x_train, y:y_train})
			self._run(sess,train,{x:x_train, y:y_train})

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
