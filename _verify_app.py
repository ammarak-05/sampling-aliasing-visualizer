from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception


def show(label):
    print(
        label,
        "exceptions=", len(at.exception),
        "f0=", at.slider(key="f0_slider").value,
        "fs=", at.slider(key="fs_slider").value,
        "mode=", at.radio(key="mode").value,
        "wave=", at.radio(key="waveform").value,
        "recon=", at.radio(key="reconstruction").value,
        "prefilter=", at.checkbox(key="prefilter").value,
    )
    if at.exception:
        for e in at.exception:
            print("EXC:", e.value)


show("initial")
at.button(key="quick_alias").click().run()
show("1.3x")
at.button(key="quick_2x").click().run()
show("2x")
at.button(key="quick_10x").click().run()
show("10x")

for wave in ["Square", "Triangular", "Sine"]:
    at.radio(key="waveform").set_value(wave).run()
    show("wave " + wave)

for method in [
    "Linear Interpolation",
    "Zero-Order Hold (ZOH)",
    "Whittaker-Shannon (Sinc)",
]:
    at.radio(key="reconstruction").set_value(method).run()
    show("recon " + method)

at.checkbox(key="prefilter").check().run()
show("prefilter on")
at.checkbox(key="prefilter").uncheck().run()
show("prefilter off")

at.radio(key="mode").set_value("Audio Synthesis (200 Hz – 4000 Hz)").run()
show("audio mode")
at.number_input(key="f0_number").set_value(4000.0).run()
show("f0 4000")
at.number_input(key="fs_number").set_value(5200.0).run()
show("fs 5200")
at.radio(key="mode").set_value("Visual Inspection (1 Hz – 50 Hz)").run()
show("visual mode")
at.slider(key="f0_slider").set_value(1.0).run()
show("f0 1")
at.slider(key="fs_slider").set_value(0.5).run()
show("fs 0.5")
print("done")
