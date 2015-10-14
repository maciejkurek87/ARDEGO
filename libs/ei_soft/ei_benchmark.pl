#!/usr/bin/perl
#use Capture::Tiny 'tee';
system("echo executing");
#start from highest precision, starting N
for($numLambda=1; $numLambda<=6; $numLambda++)
{
    $step = 10;
    $cmd = sprintf("echo %d", $numLambda);
    system($cmd);
    for($n_sims=100; $n_sims<=1000000000; $n_sims*=$step)
    {
            $cmd = sprintf("./ei_bench %d %d %d %d", $n_sims, $numLambda, $numLambda, 1.02213);
            system($cmd);
    }   
}
