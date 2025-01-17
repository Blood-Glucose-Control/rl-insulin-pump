# rl-insulin-pump

Creating RL Agents for simglucose/PadovaT1D simulator.

##### Getting Started

1. Set up virtual environment

``` bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies

``` bash
pip install -r requirements.txt
```

3. Install simglucose in editable mode:

``` bash
cd simglucose
pip install -e . --config-settings editable_mode=compat
cd ..
```

4. Run scripts from root dir
``` bash
python -m src.main
```

Installing simglucose in editable mode within the project directory allows for direct imports and potential modifications to the source code. This gives us much greater control for customization and enables effective debugging.

##### Managing the `simglucose` Fork

This repository includes `simglucose` as a git subtree. If you need to update or push changes to the [simglucose fork](https://github.com/Blood-Glucose-Control/simglucose.git), use the following commands **in the root directory**.

###### Pulling Updates from the Subtree
If the upstream `simglucose` repository has updates, you can pull them into the subtree:
```bash
git subtree pull --prefix=simglucose https://github.com/Blood-Glucose-Control/simglucose.git master --squash
```

###### Pushing Changes to the Subtree

To push changes made to `simglucose` back to the forked repository:

```bash
git subtree push --prefix=simglucose https://github.com/Blood-Glucose-Control/simglucose.git master
```
