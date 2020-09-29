import subprocess as sp

# # ok
# pipe = sp.Popen( 'dir', shell=True, stdout=sp.PIPE, stderr=sp.PIPE )
# # res = tuple (stdout, stderr)
# res = pipe.communicate()
# print("retcode =", pipe.returncode)
# print("res =", res)
# print("stderr =", res[1])
# for line in res[0].decode(encoding='utf-8').split('\n'):
#   print(line)

# with error
pipe = sp.Popen( 'ls /bing', shell=True, stdout=sp.PIPE, stderr=sp.PIPE )
res = pipe.communicate()
print("retcode =", pipe.returncode)
print("res =", res)
print("stderr =", res[1])