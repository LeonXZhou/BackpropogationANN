# BackpropogationANN
An implementation of an ANN using backpropogation to classify glass based on chemical data

## Design Consideration:

## Data PreProcessing:
all attribute data was normalized to a range of -1 to 1. This was eliminate the effect of some
attributes having higher ranges. This reduces training time required for a network to learn
the different ranges. The data was randomly split into rought 70% training 15% validation and
15% testing. The training data was used to train the network. The validation data was used to 
checking the progress of the network inbetween each training epoch and used to determine termination.
Test set is used once the network is fully trained to evaluate the training of the network. 

## Activation function:
A sigmoid fuction was used as the activation function. It was selected because it is
differentiable everywhere and its derivative allows for easy matrix manipulation allowing
faster and easier gradient descent calculations.

## Learning rate:
An initial learning rate of 0.05 was selected through expiremental trials. The learning
rate is reduced after each epoch to avoid overshooting when the neural network approaches
the minimum of the error function.

## Momentum rate:
A momentum constant of one 1 was selected. This means the previous weight changes are applied
in new weight changes. This is to avoid the network from getting stuck at a local minimum. The
value was selected experimentally.

## Training:
Over each epoch 5 trials from each class of glass in the training set were randomly selected.
The number of trials from each class was kept consistent to eliminate any bias due to frequency. 
A gaussian noise was applied to the training data to help prevent over training. The gaussian was
based on the standard deviation of each training attribute.A back propagation algorithm was used
to minimize the mean squared error between expected and predicted. At the end of each epoch a mean
squared error is computed using the validation data set. If the mean squared error was less than 1
the training is terminated to help prevent overfitting and computation waste. Otherwise the training
terminates after 10000 epochs. 

## Network Architecture:
The network has 4 layers. An input layer, 2 hidden layers, and an output layer. The input layer takes
in 10 inputs. 9 attributes of the glass and a bias set to 1. The first hidden layer has 15 nodes and
the second hidden layer as 8 nodes. These values were selected through manual pruning. Manually expierementing by 
starting at 20 nodes at each hidden layer these were manually reduced until an optimal values were reached. The final
layer has 6 outputs, one for each class of glass.



## Initial weights for first hidden layer:
0.914370 0.865082 0.046703 0.007839 0.749507 0.733540 0.708370 0.157083 0.118546 0.160402<br />
0.823173 0.403718 0.179131 0.631456 0.038122 0.094103 0.111688 0.689453 0.617266 0.573803<br />
0.826109 0.744778 0.226651 0.087889 0.500936 0.313692 0.267973 0.509442 0.701629 0.907184<br />
0.453793 0.533209 0.808815 0.389225 0.742982 0.984910 0.210742 0.578415 0.720718 0.133563<br />
0.813009 0.737116 0.744398 0.843540 0.749759 0.741114 0.641905 0.639724 0.827852 0.834031<br />
0.221491 0.662176 0.179220 0.821547 0.031830 0.034246 0.019430 0.645580 0.531656 0.027561<br />
0.026443 0.747682 0.748696 0.231421 0.784528 0.844238 0.830473 0.555103 0.691804 0.772080<br />
0.738744 0.407069 0.671334 0.696927 0.815064 0.349329 0.523727 0.191797 0.252445 0.541462<br />
0.301499 0.645023 0.048902 0.109378 0.757677 0.442853 0.969704 0.122535 0.357824 0.923496<br />
0.835875 0.978164 0.195541 0.359358 0.645231 0.277071 0.046883 0.581562 0.187649 0.789332<br />
0.380350 0.326462 0.651303 0.624781 0.311576 0.747926 0.403190 0.764965 0.044218 0.909848<br />
0.646363 0.452632 0.022045 0.208820 0.613291 0.928056 0.604249 0.561443 0.414324 0.835874<br />
0.581273 0.744619 0.245759 0.114409 0.989778 0.032584 0.221483 0.415283 0.239269 0.265486<br />
0.555761 0.609951 0.606754 0.761457 0.215281 0.893572 0.660373 0.045241 0.596884 0.377381<br />
0.344302 0.198761 0.715419 0.977297 0.277856 0.655494 0.437237 0.433141 0.105746 0.251500<br />

## Initial weights for second hidden layer:
0.868166 0.115874 0.316558 0.714881 0.379436 0.270454 0.946904 0.910009 0.610268 0.996166<br />
0.986600 0.877955 0.501476 0.811653 0.795903 0.177586 0.607412 0.097878 0.228821 0.143242<br />
0.309567 0.453432 0.296381 0.559509 0.871787 0.207077 0.057601 0.908057 0.340157 0.841250<br />
0.932335 0.809508 0.062756 0.402848 0.256758 0.644094 0.572783 0.925574 0.393879 0.837642<br />
0.682155 0.333715 0.284062 0.705207 0.571306 0.971318 0.219736 0.235971 0.693739 0.661440<br />
0.919865 0.921976 0.253152 0.537469 0.395532 0.091576 0.273092 0.085942 0.976316 0.899857<br />
0.833804 0.979417 0.597429 0.679840 0.221631 0.926042 0.038522 0.632132 0.039706 0.973365<br />
0.875600 0.503736 0.589374 0.280825 0.153115 0.829326 0.131340 0.132213 0.344281 0.988158<br />
0.483284 0.307517 0.250081 0.781515 0.146196 0.438449 0.868073 0.013910 0.757433 0.979000<br />
0.233058 0.740665 0.800296 0.871220 0.696047 0.221020 0.081357 0.559141 0.559403 0.399655<br />
0.545971 0.166365 0.173385 0.714783 0.996790 0.800980 0.166106 0.773925 0.290416 0.927493<br />
0.473025 0.398022 0.099549 0.505041 0.088047 0.684744 0.137701 0.470065 0.247515 0.999498<br />

## Initial weights for output layer:
0.272487 0.585166 0.014404 0.611100 0.510928 0.482799 0.402844 0.889099<br />
0.183305 0.693591 0.777610 0.734366 0.606952 0.700405 0.240744 0.085129<br />
0.910373 0.535635 0.241266 0.843579 0.816082 0.238501 0.872122 0.326394<br />
0.382095 0.404967 0.506162 0.240476 0.473857 0.261755 0.391146 0.887700<br />
0.343029 0.143025 0.725594 0.287595 0.773354 0.996943 0.558059 0.172671<br />
0.435048 0.411911 0.015288 0.076989 0.688242 0.549667 0.932455 0.594740<br />

## Final weights for first hidden layer:
2.847682 1.100412 -2.160635 1.059413 0.741289 0.569782 1.786235 1.080982 -0.206226 0.005828 1.024981 0.517612 -0.046032 -0.481627 0.214648<br />
-0.628380 0.213792 1.830872 2.051265 0.084711 0.987389 -1.490409 0.827241 1.155031 -0.404622 2.069585 0.004111 -0.894358 0.128909 1.655919<br />
1.648353 0.951385 0.690623 -1.234911 0.685768 0.533183 0.209336 0.629966 1.530169 0.127212 3.237338 -1.039963 0.002983 0.069010 1.894004<br />
1.364090 0.224166 1.159949 -0.286195 0.203148 0.023472 1.451506 -1.148174 1.506007 1.555429 -0.340787 -0.683499 0.576934 0.250450 -0.440035<br />
-1.021568 0.483518 1.448182 -0.562482 -0.619558 0.628277 2.158536 0.211258 2.121205 0.317956 0.496196 1.358114 1.636761 0.011059 -0.342970<br />
-0.410065 1.035909 -1.609888 -0.756020 1.592212 0.816610 0.231225 -0.961278 0.848032 1.001489 0.267336 2.730682 -0.221782 0.585047 1.919751<br />
0.071448 2.381292 1.667271 0.450938 1.755254 -0.216630 -1.748013 -0.728361 -0.998727 2.376762 -0.218098 -0.159364 0.353766 2.114745 0.507301<br />
1.657661 -1.382917 0.791262 1.446292 0.731529 0.641926 1.375623 -1.341402 1.869320 0.183444 0.196772 1.469000 0.895393 0.243449 1.075221<br />
1.062211 2.399365 -0.255962 -0.916323 1.764621 0.666600 -0.280894 1.452389 -0.273858 -0.727000 0.081696 1.706542 0.127167 1.722110 0.614267<br />
-0.053012 0.141202 -0.159263 0.226743 0.151343 2.531614 0.606612 1.024730 0.944231 0.493100 0.520807 -2.525746 2.274973 -1.415898 -0.221900<br />

## Final weights for second hidden layer:
0.902669 0.195322 0.429790 0.788215 0.411138 0.294428 1.118602 1.083737 <br />
0.714132 1.169778 1.051776 0.945982 0.610961 0.873255 0.815479 -0.074083 <br />
1.241193 -1.818012 0.060486 0.489558 -1.169036 -0.010453 -2.198802 1.385631 <br />
3.000144 -0.894971 -1.410272 3.312440 -1.550514 1.119136 0.979621 0.837129 <br />
0.118397 0.449603 0.273853 0.696132 0.638507 1.032139 0.454550 0.907305 <br />
0.761989 0.413851 0.310976 0.770961 0.620020 1.063326 0.415015 0.496675 <br />
0.899226 0.800746 1.002862 1.250823 0.599415 0.762996 0.729380 0.326897 <br />
0.452169 0.306214 1.150163 1.026166 0.808750 1.007537 0.648440 0.710272 <br />
0.184130 0.913115 0.162187 0.758240 0.073349 1.108093 0.855989 0.497827 <br />
0.657196 0.272785 0.106568 3.624744 -1.648091 -2.593607 -1.269185 0.933678 <br />
2.508252 -2.849038 -2.338954 1.424965 -0.583028 -1.190640 1.647349 0.769331 <br />
1.228205 0.802924 -0.423639 0.904275 0.456144 1.257464 2.003154 -0.766429 <br />
-0.585211 -0.362933 -0.198646 -0.589073 -0.379199 -1.325324 -0.204201 -0.782585 <br />
1.075592 1.815879 1.615868 1.234436 1.568131 3.077230 1.488762 -1.868630 <br />
-1.636247 -2.075121 -3.098209 2.550621 -1.203267 -0.560583 -0.085784 4.528541 <br />

## Final weights for output layer:
-0.756688 0.791223 -1.062115 -0.423515 -0.553253 -4.083479 <br />
2.052543 1.042193 -0.714028 -2.493372 -0.101419 -0.163543 <br />
-0.265403 -4.182829 -0.017206 3.389711 1.129890 0.190763 <br />
0.079788 0.709339 0.551001 -5.234499 -1.110027 -6.073881 <br />
-1.520384 -6.692250 -1.290069 -1.510671 -1.381707 3.825034 <br />
-0.931439 5.563477 -0.867874 0.394510 -0.335713 -0.806179 <br />
-0.227642 5.733064 -1.924936 -6.748556 -2.754218 6.166466 <br />
-2.854686 -2.879711 -2.052195 2.124023 -0.968644 5.091148 <br />

## Confusion Matrix:

p	  Expected<br /> 
r	7 2  0 0 0 0<br />
e	3 10 1 0 0 0<br />
d	1 1  1 0 0 0<br />
i	0 0  0 0 0 1<br />
c	0 0  0 0 3 0<br />
t	0 0  0 0 0 9<br />

Recall: 0.637566<br />
Precision: 0.634266<br />

## TEST DATA OUTPUT
Expected       Predicted<br />
1              1<br />
1              1<br />
1              1<br />
1              1<br />
1              1<br />
1              2<br />
2              2<br />
2              2<br />
2              2<br />
2              2<br />
2              2<br />
2              2<br />
3              3<br />
5              5<br />
5              6<br />
6              6<br />
6              6<br />
6              6<br />
6              6<br />
6              2<br />
1              1<br />
2              1<br />
2              3<br />
1              1<br />
1              1<br />
2              3<br />
2              2<br />
2              2<br />
2              2<br />
2              1<br />
2              1<br />
2              3<br />
2              6<br />
6              6<br />
4              7<br />
7              7<br />
7              7<br />
7              7<br />
7              7<br />
