#!/usr/bin/env bash
set -x

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)        Build"
    echo "  `basename $0` (r | run)          Run"
    echo "  `basename $0` (p | push)         Push"
    echo "  `basename $0` (d | rm)           Remove Container"
    echo "  `basename $0` (s | stop)         Stop"
    echo "  `basename $0` (k | kill)         Kill"
    echo "  `basename $0` rm                 Remove"
    echo
    echo "  `basename $0` (l | log)                 Show log tail (last 100 lines)"
    echo "  `basename $0` (e | exec)     <command>  Execute command"
    echo "  `basename $0` (a | attach)              Attach to container with shell"
    echo
    echo "Arguments:"
    echo
    echo "  command       Command to be executed inside a container"
    exit
fi
# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file with USERNAME and PASSWORD for the Learning Loop"

cmd=$1
cmd_args=${@:2}

image_name="yolo_tensorrt_converter"
container_name='yolo_tensorrt_converter'

run_args=""
run_args+="-v $(pwd)/converter:/app "
run_args+="-v $HOME/data:/data "
# run_args+="-v $HOME/learning_loop_node/learning_loop_node:/usr/local/lib/python3.7/dist-packages/learning_loop_node "
run_args+="-e HOST=learning-loop.ai "
run_args+="-e USERNAME=$USERNAME -e PASSWORD=$PASSWORD "
run_args+="--name $container_name "

case $cmd in
    b | build)
        docker kill $container_name
        docker rm $container_name # remove existing container
        docker build . -t $image_name $cmd_args
        docker build . -t ${image_name}-dev $cmd_args
        ;;
    r | run)
	    nvidia-docker run -it --rm $run_args ${image_name}-dev $cmd_args
        ;;
    p | push)
        docker push ${image_name}-dev 
        docker push $image_name
        ;;
    s | stop)
        docker stop $container_name $cmd_args
        ;;
    k | kill)
        docker kill $container_name $cmd_args
        ;;
    d | rm)
        docker kill $container_name
        docker rm $container_name $cmd_args
        ;;
    l | log | logs)
        docker logs -f --tail 100 $cmd_args $container_name
        ;;
    e | exec)
        docker exec $cmd_args $container_name
        ;;
    a | attach)
        docker exec -it $cmd_args $container_name /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac

