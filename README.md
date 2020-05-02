
# AutoBlur

Blur faces in a video automatically.

## Motivation

Often the need arises to blur/pixelate the faces of individuals in a video to protect their privacy. Often this is done in a manual fashion on a frame by frame basis. With AutoBlur, we leverage the power of deep learning to build something potentially useful and easy to use!

## Insatallation

**Note:** The default *master* branch comes with many demo videos. This increases the repository size unnecessarily. Hence, for a lighter implementation devoid of these videos, clone the *light* branch.

To get the default *master* branch:

```
git clone https://github.com/pranjaldatta/AutoBlur.git --single-branch --branch master
```

To get the *light* branch:

```
git clone https://github.com/pranjaldatta/AutoBlur.git --single-branch --branch light
```

## Usage

For Usage, please check out [pixel_demo.py](https://github.com/pranjaldatta/AutoBlur/blob/master/pixel_demo.py)

## Next Steps

Often the intention is to blur/pixelate out all faces except one or more particular faces. Hence, it would be a great feature if the user could be provided a GUI based interface wherein he can select faces to be **left out** of pixelation in real-time.

- [ ] Construct a GUI interface to allow the user to **leave out** faces from pixelation.
