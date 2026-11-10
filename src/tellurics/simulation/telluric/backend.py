import warnings

from tellurics.simulation.utils.telfit_home import add_telfit_to_path


class TelFitBackend:
    def __init__(self, modeler=None):
        self._modeler = modeler

    def _get_modeler(self):
        if self._modeler is None:
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
            )
            warnings.filterwarnings(
                "ignore",
                message="PYSYN_CDBS is undefined.*",
            )
            warnings.filterwarnings(
                "ignore",
                message="Extinction files not found.*",
            )

            add_telfit_to_path()
            from telfit import Modeler

            self._modeler = Modeler(
                print_lblrtm_output=False,
                debug=False,
            )

        return self._modeler

    def make_model(
        self,
        params,
        *,
        lowfreq,
        highfreq,
        observatory=None,
        workdir=None,
    ):
        modeler = self._get_modeler()
        workdir = str(workdir) if workdir is not None else None

        if observatory is None:
            return modeler.MakeModel(
                lowfreq=lowfreq,
                highfreq=highfreq,
                workdir=workdir,
                vac2air=False,
                libfile="",
                save=False,
                **params,
            )

        return modeler.MakeModel(
            lowfreq=lowfreq,
            highfreq=highfreq,
            lat=observatory["lat"],
            alt=observatory["alt"],
            workdir=workdir,
            vac2air=False,
            libfile="",
            save=False,
            **params,
        )
