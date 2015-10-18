sub modify {
    my $tag = $_[0];
    my $newtag = $_[1];
    my $file = $_[2];
    system("sed -i '/$tag/c\\$newtag' ".$file);
}

sub time_since_epoch { return `date +%s` }

$app = 'xinyu_rtm';
$run_dir = 'examples/xinyu_rtm/'; 
$config_file = $run_dir."configuration_script.py";

$fitness_file = $run_dir."fitness_script.py";

$results_folder = "/data/mk306/hc_rtm";
#system("rm -Rf $results_folder");
$parallelism = 1;
system("mkdir ".$results_folder); 
$pid = 1;
for ($i=0; $i < $parallelism - 1; $i++) {
    if ($pid) {
        $pid = fork();
    }
    else { 
        last;
    }
}
if ($pid) {
    $i = $parallelism;
}
sleep($i * 5);
$epoch = time_since_epoch();
chomp($epoch);
$tempfile = "/data/mk306/temp$epoch.txt";
$tempfolder = '/data/mk306/mk306/hc_'.$epoch.'/';
system('mkdir '.$tempfolder); 
system("cp -Rf * $tempfolder");
chdir $tempfolder;
system("echo `pwd`");
##modify the configuration file to pso with trails_count
modify("results_folder_path =", 'results_folder_path ="'.$results_folder.'"', $config_file);
modify("trials_type =", 'trials_type =\"Gradient_Trial\"', $config_file);
modify("surrogate_type =", 'surrogate_type ="bayes2"', $config_file);
modify("trials_count =", "trials_count = 20", $config_file);

$j = 0;
#@maxError = ('maxError = 0.1', 'maxError = 0.01', 'maxError = 0.001'); 
if (($j % $parallelism) == ($i - 1)){ #easiest way to divide tasks
    #print $i."\n";
    #print "make run_example2 &> $tempfile\n";
    print "start $j $maxErrorSet $max_stdv \n";
    system("make run_example4 &> $tempfile");
    print "done $j\n";
}
$j = $j + 1;

