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
	def Variable(self,value):
		return tf.Variable(value, dtype=tf.float32)
	def placeholder(self):
		return tf.placeholder(tf.float32)
	def square(self,value):
		return tf.square(value)
	def reduce_sum(self,value):
		return tf.reduce_sum(value)
	def global_variables_initializer(self):
		return tf.global_variables_initializer()
	def Session(self):
		return tf.Session()
	def run(self,x,y,x_train,y_train,sess):
		# Model parameters
#		W = tf.Variable([.3], dtype=tf.float32)
		W = self.Variable([.3])
#		b = tf.Variable([-.3], dtype=tf.float32)
		b = self.Variable([-.3])

		# Model input and output
#		x = tf.placeholder(tf.float32)
#		x = self.placeholder()
		linear_model = W * x + b
#		y = tf.placeholder(tf.float32)
#		y = self.placeholder()

		# loss
#		loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
#		loss = tf.reduce_sum(self.square(linear_model - y)) # sum of the squares
		loss = self.reduce_sum(self.square(linear_model - y)) # sum of the squares

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		train = optimizer.minimize(loss)

		# training data
#		x_train = [1,2,3,4]
#		y_train = [0,-1,-2,-3]

		# training loop
#		init = tf.global_variables_initializer()
		init = self.global_variables_initializer()
#		sess = tf.Session()
#		sess = self.Session()
		sess.run(init) # reset values to wrong
		for i in range(1000):
		  sess.run(train, {x:x_train, y:y_train})

		# evaluate training accuracy
		curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
		print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
_END_

		my $foo = $ip.call('__main__', 'TensorFlow');
		$foo.run(
			x => $foo.placeholder(),
			y => $foo.placeholder(),
			x_train => [ 1, 2, 3, 4 ],
			y_train => [ 0, -1, -2, -3 ],
			sess => $foo.Session()
		);
	}
}
