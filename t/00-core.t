use v6;

use Test;

use TensorFlow;
use TensorFlow::NumPy;
use TensorFlow::Python;

my $tf = TensorFlow.new;
my $np = TensorFlow::NumPy.new( python => $tf.python );
my $py = TensorFlow::Python.new( python => $tf.python );

#
# import tensorflow as tf
# def return_tf_object()
#   retun tf.float32
#

subtest {

	# Model Parameters
	#
	my $W = $tf.Variable( [ 0.3 ], dtype => $tf.float32 );
	my $b = $tf.Variable( [ -0.3 ], dtype => $tf.float32 );

	# Model input and output
	#
	my $x = $tf.placeholder( $tf.float32 );
	my $linear_model = $W * $x + $b;

	my $y = $tf.placeholder( $tf.float32 );

	# loss
	
	my $loss = $tf.reduce_sum( $tf.square( $linear_model - $y ));

	# optimizer
	#
	my $optimizer = $tf.train.GradientDescentOptimizer( 0.01 );
	my $train = $optimizer.minimize( $loss );

	# training data
	#
	my $x_train = [ 1, 2, 3, 4 ];
	my $y_train = [ 0, -1, -2, -3 ];

	# training loop
	#
	my $init = $tf.global_variables_initializer();
	my $sess = $tf.Session();
	$sess.run( $init );

	for ^1000 {
		$sess.run( $train, [ $x, $x_train,  $y, $y_train ] );
	}

	my ( $curr_W, $curr_b, $curr_loss ) = $sess.run(
		[ $W, $b, $loss ],
		[ $x, $x_train, $y, $y_train ]
	);
	$py.print("W: %s b: %s loss: %s", $curr_W, $curr_b, $curr_loss );

#`(
$tf.python.run( Q:to[_END_] );
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
_END_
)

	done-testing;
}, 'getting started';

subtest {

#`(
	my $features = [
		$tf.contrib.layers.real_valued_column(
			'x', dimensions => 1
		)
	];
	my $estimator = $tf.contrib.learn.LinearRegressor(
		feature_columns => $features
	);

	my $x_train = $np.array( [ 1.0, 2.0, 3.0, 4.0 ] );
	my $y_train = $np.array( [ 0.0, -1.0, -2.0, -3.0 ] );
	my $x_eval = $np.array( [ 2.0, 5.0, 8,0, 1.0 ] );
	my $y_eval = $np.array( [ -1.01, -4.1, -7.0, 0.0 ] );

	my $input_fn = $tf.contrib.learn.io.numpy_input_fn(
		[ 'x', $x_train ],
		$y_train,
		batch_size => 4,
		num_epochs => 1000
	);

	my $eval_input_fn = $tf.contrib.learn.io.numpy_input_fn(
		[ 'x', $x_eval ],
		$y_eval,
		batch_size => 4,
		num_epochs => 1000
	);

	$estimator.fit( input_fn => $input_fn, steps => 1000 );

	my $train_loss = $estimator.evaluate( input_fn => $input_fn );
	my $eval_loss = $estimator.evaluate( input_fn => $eval_input_fn );
	$py.print("train loss: %r", $train_loss );
	$py.print("eval loss: %r", $eval_loss );
)
	
#`(
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
)

	done-testing;
}, 'getting started 2';

subtest {

#`(
	sub model( $features, $labels, $mode ) {
		# Build a linear model and predict values
		my $W = $tf.get_variable("W", [ 1 ], dtype => $tf.float64 );
		my $b = $tf.get_variable("b", [ 1 ], dtype => $tf.float64 );
		my $y = $W * $features{'x'} + $b;
	
		# Loss sub-graph
		my $loss = $tf.reduce_sum( $tf.square( $y - $labels ) );

		# Training sub-graph
		my $global_step = $tf.rain.get_global_step();
		my $optimizer = $tf.train.GradientDescnOptimizer( 0.01 );
		my $train = $tf.group(
			$optimizer.minimize( $loss ),
			$tf.assign_add( $global_step, 1 )
		);

		# ModelFnOps...
		return $tf.contrib.learn.ModelFnOps(
			mode => $mode,
			predictions => $y,
			loss => $loss,
			train_op => $train
		);
	}

	my $estimator = $tf.contrib.learn.Estimator(
		model_fn => &model
	);

	# Define our data sets
	my $x_train = $np.array( [ 1.0, 2.0, 3.0, 4.0 ] );
	my $y_train = $np.array( [ 0.0, -1.0, -2.0, -3.0 ] );
	my $x_eval = $np.array( [ 2.0, 5.0, 8.0, 1.0 ] );
	my $y_eval = $np.array( [ -1.01, -4.1, -7.0, 0.0 ] );
	my $input_fn = $tf.contrib.learn.io.numpy_input_fn(
		{ 'x' => $x_train },
		$y_train,
		4,
		num_epochs => 1000
	);
	my $eval_input_fn = $tf.contrib.learn.io.numpy_input_fn(
		{ x => $x_eval },
		$y_eval,
		batch_size => 4,
		num_epochs => 1000
	);

	# train
	$estimator.fit( input_fn => $input_fn, steps => 1000 );

	my $train_loss = $estimator.evaluate( input_fn => $input_fn );
	my $eval_loss = $estimator.evaluate( input_fn => $eval_input_fn );
	$py.print( "train loss: %r", $train_loss );
	$py.print( "eval loss: %r", $eval_loss );
)

#`(
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7.0, 8.0])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did. 
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
)

	done-testing;
}, 'getting started 3';

done-testing;

# vim: ft=perl6
